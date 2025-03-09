import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import tempfile
import time
import google.generativeai as genai
from langchain.text_splitter import NLTKTextSplitter
from langchain.text_splitter import TokenTextSplitter
import numpy as np

# 加载预训练的句子嵌入模型

def semantic_chunking(text, embedding_model, chunk_size=5, similarity_threshold=0.75):
    """
    按语义相似度切割文本，确保每个 chunk 内部语义相关。
    :param chunk_size: 每个 chunk 最多包含的句子数。
    :param similarity_threshold: 语义相似度的阈值（越高分块越细）。
    :return: 语义分块后的文本列表。
    """
    # 先按句子拆分，句号分割
    sentences = text.split(". ") 
    
    # 计算句子的 embedding
    embeddings = [embedding_model.embed_query(sentence) for sentence in sentences]

    chunks = []
    current_chunk = [sentences[0]]  # 初始化第一个 chunk
    current_embedding = embeddings[0]  # 取第一个句子的 embedding
    
    for i in range(1, len(sentences)):
        similarity = np.dot(current_embedding, embeddings[i]) / (np.linalg.norm(current_embedding) * np.linalg.norm(embeddings[i]))
        
        if len(current_chunk) >= chunk_size or similarity < similarity_threshold:
            # 开始新的 chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_embedding = embeddings[i]
        else:
            # 继续加入当前 chunk
            current_chunk.append(sentences[i])
            current_embedding = np.mean([current_embedding, embeddings[i]], axis=0)  # 取平均 embedding
    
    # 添加最后一个 chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Set Google API key (replace with your key or use an env variable)
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(page_title="Chat with Your PDFs (Gemini)")

st.title("📄💬 Chat with Your PDFs (Gemini)")

# File uploader
uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    documents = []
    
    # Process PDFs if vector store doesn't exist in session state
    if "vector_store" not in st.session_state:
        with st.spinner("Processing your PDFs..."):
            # Create a temporary directory to save the uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    # Save the uploaded file to a temporary file
                    temp_file_path = os.path.join(temp_dir, file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Load the PDF
                    loader = PyPDFLoader(temp_file_path)
                    documents.extend(loader.load())

                # Split documents into chunks
                #embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                #text_splitter = SemanticChunker(embedding_model, chunk_size=1000)
                #text_splitter = NLTKTextSplitter(chunk_size=500)
                #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

                # Generate embeddings and store in FAISS
                # ------------------------------------------------- #
                embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

                # **运行语义分块**
                print(type(documents[1]))
                docs = semantic_chunking(documents[1].page_content, embedding_model, chunk_size=5, similarity_threshold=0.75)
                text_splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=50) 
                docs2 = text_splitter.split_documents(documents)
                print(docs[0])
                print("---")
                print(docs2[0])

                # use your embeddings model here
                st.session_state.vector_store = FAISS.from_documents(docs, embedding_model)
                # ------------------------------------------------- #


        st.success("✅ PDFs uploaded and processed! You can now start chatting.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask a question about your PDFs...")
    
    if user_input:
        # Immediately add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display the user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        
        
        # Configure retriever with more advanced parameters
        retriever = st.session_state.vector_store.as_retriever(
            #search_type="similarity_score_threshold",
            #search_kwargs={"k": 10, "score_threshold": 0.5}
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.6}  # Adjust these parameters as needed
        )


        
        # Define a custom prompt template for the chatbot
        # from langchain.prompts import PromptTemplate

        # Create a custom prompt template
        template = """You are a helpful assistant that answers questions based on the provided documents.
        
        Given the context information and not prior knowledge, answer the following question:
        Question: {user_input}
        
        Answer the question with detailed information from the documents. If the answer is not in the documents, 
        say "I don't have enough information to answer this question." Cite specific parts of the documents when possible.
        """
        
        # Create the QA chain with the custom prompt
        # The RetrievalQA chain will automatically handle getting the context from the retriever
        # and formatting it with the prompt template
        qa_chain = RetrievalQA.from_chain_type(
            ## use your llm model here
            #llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5),
            llm=ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05", temperature=0.2),
            retriever=retriever,
            chain_type="stuff",  # "stuff" chain type puts all retrieved documents into the prompt context
            return_source_documents=True,  # Return source documents for reference
            verbose = True,
            chain_type_kwargs={
                # "prompt": CUSTOM_PROMPT,  # Use the custom prompt
                "verbose": True  # Enable verbose mode to see the full prompt
            }
        )
        # ------------------------------------------------- #
        
        # Get response from the chatbot with spinner
        with st.spinner("Thinking..."):
            # The RetrievalQA chain automatically:
            # 1. Takes the query
            # 2. Retrieves relevant documents using the retriever
            # 3. Formats those documents as the context in the prompt
            # 4. Sends the formatted prompt to the LLM
            response = qa_chain.invoke({"query": template.format(user_input =user_input)})
            
            # For debugging, you can see what's in the response
            # st.write("Response keys:", list(response.keys()))
            
            # Display retrieved chunks in an expander if source documents are available
            if "source_documents" in response:
                with st.expander("View Retrieved Chunks (Context)"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Chunk {i+1}**")
                        st.markdown(f"**Content:** {doc.page_content}")
                        st.markdown(f"**Source:** Page {doc.metadata.get('page', 'unknown')}")
                        st.markdown("---")
            
            response_text = response["result"]
        # Display assistant response
        # with st.chat_message("assistant"):
        #     st.markdown(response_text)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Simulate streaming with an existing string
            for chunk in response_text.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response)
                time.sleep(0.05)  # Small delay to simulate streaming
                
        # Store assistant response in session state
        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.info("Please upload PDF files to begin.")
