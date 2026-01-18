import os
import torch
import logging
import streamlit as st
import duckdb
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration

# Configure logging to display information level messages
logging.basicConfig(level=logging.INFO)

# Constants for model and database paths
MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
MODEL_PATH = (
    "/home/lloyd/Downloads/local_model_store/microsoft/Phi-3-mini-128k-instruct"
)
DB_PATH = "/home/lloyd/Downloads/local_model_store/phi3t5_conversation_embeddings.db"
EMBEDDING_MODEL_NAME = "t5-small"
EMBEDDING_MODEL_PATH = "/home/lloyd/Downloads/local_model_store/t5-small"

# Determine the best device for computation, using CPU for compatibility with lightweight devices
device = torch.device("cpu")

# Initialize a database connection for storing conversation embeddings
conn = duckdb.connect(database=DB_PATH, read_only=False)
conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_id START 1")
conn.execute(
    "CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY DEFAULT nextval('seq_id'), input TEXT, output TEXT, input_embedding BLOB, output_embedding BLOB)"
)


@st.cache(allow_output_mutation=True)
def load_chat_model(model_path: str, model_name: str):
    """
    Load or download the chat model based on availability in the local storage.

    Args:
    model_path (str): The path where the model is stored or will be stored.
    model_name (str): The name of the model to be used for downloading if not available locally.

    Returns:
    AutoModelForCausalLM: The loaded chat model.
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        model.save_pretrained(model_path)
        logging.info("Chat model downloaded and saved.")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        logging.info("Chat model loaded from local storage.")
    model.to(device)
    logging.info(f"Chat model moved to device: {device}")
    return model


@st.cache(allow_output_mutation=True)
def load_embedding_model(embedding_model_path: str, embedding_model_name: str):
    """
    Load or download the embedding model based on availability in the local storage.

    Args:
    embedding_model_path (str): The path where the embedding model is stored or will be stored.
    embedding_model_name (str): The name of the model to be used for downloading if not available locally.

    Returns:
    T5ForConditionalGeneration: The loaded embedding model.
    """
    if not os.path.exists(embedding_model_path):
        os.makedirs(embedding_model_path, exist_ok=True)
        embedding_model = T5ForConditionalGeneration.from_pretrained(
            embedding_model_name
        )
        embedding_model.save_pretrained(embedding_model_path)
        logging.info("Embedding model downloaded and saved.")
    else:
        embedding_model = T5ForConditionalGeneration.from_pretrained(
            embedding_model_path
        )
        logging.info("Embedding model loaded from local storage.")
    embedding_model.to(device)
    logging.info(f"Embedding model moved to device: {device}")
    return embedding_model


# Load models using the defined functions
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
model = load_chat_model(MODEL_PATH, MODEL_NAME)
embedding_model = load_embedding_model(EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME)


# Function to generate text based on a given prompt
def generate_text(prompt: str) -> str:
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    logging.info("Input encoded and moved to device.")
    try:
        outputs = model.generate(
            inputs,
            max_new_tokens=500,
            top_p=0.95,
            do_sample=True,
            top_k=60,
            temperature=0.95,
            early_stopping=True,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info("Text generation successful.")
        return generated_text
    except Exception as e:
        logging.error(f"Error during text generation: {e}")
        return "Error in text generation."


# Function to convert text to embeddings using the embedding model
def text_to_embeddings(text: str):
    embeddings_input = embedding_tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = (
            embedding_model.encoder(embeddings_input)
            .last_hidden_state.mean(dim=1)
            .cpu()
            .numpy()
        )
    return embeddings


# Setup the Streamlit app configuration
st.set_page_config(page_title="Interactive Text Generation with Phi-3", layout="wide")
st.title("Interactive Text Generation with Phi-3")

# Create two columns for the chat interface
col1, col2 = st.columns([2, 1])

# Retrieve conversation history from the database
conversation_history = conn.execute("SELECT input, output FROM embeddings").fetchall()

with col2:
    st.header("Conversation Log")
    conversation_container = st.empty()

# Display the conversation history in the Streamlit app
with conversation_container.container():
    for input_text, output_text in conversation_history:
        st.markdown(f"**User:** {input_text}")
        st.markdown(f"**Assistant:** {output_text}")
        st.markdown("---")

# Handle user interaction for text generation
with col1:
    with st.form("text_generation_form"):
        user_input = st.text_input("Enter your prompt:", key="user_input")
        submitted = st.form_submit_button("Submit")
        if submitted and user_input:
            generated_text = generate_text(user_input)
            st.markdown(f"**Assistant:** {generated_text}")

            # Generate embeddings for the input and output text, and store them in the database
            input_embeddings = text_to_embeddings(user_input)
            output_embeddings = text_to_embeddings(generated_text)

            # Minimize conversions by handling data efficiently
            input_embeddings_bytes = input_embeddings.tobytes()
            output_embeddings_bytes = output_embeddings.tobytes()

            # Store and retrieve efficiently
            conn.execute(
                "INSERT INTO embeddings (input, output, input_embedding, output_embedding) VALUES (?, ?, ?, ?)",
                [
                    user_input,
                    generated_text,
                    input_embeddings_bytes,
                    output_embeddings_bytes,
                ],
            )
            logging.info("Conversation embeddings stored.")

            # Update the conversation history in the Streamlit app
            conversation_container.empty()
            with conversation_container.container():
                for input_text, output_text in conn.execute(
                    "SELECT input, output FROM embeddings"
                ).fetchall():
                    st.markdown(f"**User:** {input_text}")
                    st.markdown(f"**Assistant:** {output_text}")
                    st.markdown("---")

# Instructions to run the Streamlit app on a local machine
# streamlit run /home/lloyd/Downloads/PythonScripts/localphi3llmtextgeneration.py --server.port 8501 --server.address 0.0.0.0
