import os
import torch
import logging
import streamlit as st
import duckdb
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from logging.handlers import RotatingFileHandler

LOG_FOLDER = "/home/lloyd/Downloads/local_model_store/logs"
LOG_FILE = os.path.join(LOG_FOLDER, "phi3t5chat.log")

# Ensure the log directory exists
os.makedirs(LOG_FOLDER, exist_ok=True)

# Setup advanced logging with a rotating file handler
log_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
log_handler = RotatingFileHandler(
    LOG_FILE,
    mode="a",
    maxBytes=10 * 1024 * 1024,
    backupCount=10,
    encoding="utf-8",
    delay=False,
)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger("phi3t5chat")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
# Constants for model and database paths
MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
MODEL_PATH = (
    "/home/lloyd/Downloads/local_model_store/microsoft/Phi-3-mini-128k-instruct"
)
DB_PATH = (
    "/home/lloyd/Downloads/local_model_store/data/phi3t5_conversation_embeddings.db"
)
EMBEDDING_MODEL_NAME = "t5-small"
EMBEDDING_MODEL_PATH = "/home/lloyd/Downloads/local_model_store/t5-small"

# Determine the best device for computation, using CPU for compatibility with lightweight devices
device = torch.device("cpu")
logger.info("Computation device set to CPU.")

# Initialize a database connection for storing conversation embeddings
try:
    conn = duckdb.connect(DB_PATH)
    logger.info("Database connected successfully.")
except duckdb.Error as db_err:
    logger.error(f"Database connection failed: {db_err}")
    st.error("Failed to connect to the database. Please check the logs.")

try:
    # Database operations
    with conn:
        conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_id START 1")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY DEFAULT nextval('seq_id'), input TEXT, output TEXT, input_embedding BLOB, output_embedding BLOB)"
        )
        logger.info("Database tables and sequences initialized.")
except duckdb.Error as db_err:
    logger.error(f"Database operation failed: {db_err}")
    st.error("An error occurred during database operations. Please check the logs.")
finally:
    conn.close()
    logger.info("Database connection closed.")

# Set Streamlit page configuration
st.set_page_config(page_title="Interactive Text Generation with Phi-3", layout="wide")


@st.cache_data
def load_chat_model(model_path: str, model_name: str):
    """
    Load or download the chat model based on availability in the local storage.

    Args:
        model_path (str): The path where the model is stored or will be stored.
        model_name (str): The name of the model to be used for downloading if not available locally.

    Returns:
        AutoModelForCausalLM: The loaded chat model.
    """
    try:
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True
            )
            model.save_pretrained(model_path)
            logger.info("Chat model downloaded and saved.")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True
            )
            logger.info("Chat model loaded from local storage.")
        model.to(device)
        logger.info(f"Chat model moved to device: {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load chat model: {e}")
        raise


@st.cache_data
def load_embedding_model(embedding_model_path: str, embedding_model_name: str):
    """
    Load or download the embedding model based on availability in the local storage.

    Args:
        embedding_model_path (str): The path where the embedding model is stored or will be stored.
        embedding_model_name (str): The name of the model to be used for downloading if not available locally.

    Returns:
        T5ForConditionalGeneration: The loaded embedding model.
    """
    try:
        if not os.path.exists(embedding_model_path):
            os.makedirs(embedding_model_path, exist_ok=True)
            embedding_model = T5ForConditionalGeneration.from_pretrained(
                embedding_model_name
            )
            embedding_model.save_pretrained(embedding_model_path)
            logger.info("Embedding model downloaded and saved.")
        else:
            embedding_model = T5ForConditionalGeneration.from_pretrained(
                embedding_model_path
            )
            logger.info("Embedding model loaded from local storage.")
        embedding_model.to(device)
        logger.info(f"Embedding model moved to device: {device}")
        return embedding_model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


# Load models using the defined functions
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
model = load_chat_model(MODEL_PATH, MODEL_NAME)
embedding_model = load_embedding_model(EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME)


# Function to generate text based on a given prompt
def generate_text(prompt: str) -> str:
    """
    Generate text based on the given prompt using the loaded model.

    Args:
        prompt (str): The text prompt for text generation.

    Returns:
        str: The generated text.
    """
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        logger.info("Input encoded and moved to device.")
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
        logger.info("Text generation successful.")
        return generated_text
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        return "Error in text generation."


# Function to convert text to embeddings using the embedding model
def text_to_embeddings(text: str):
    """
    Convert text to embeddings using the loaded embedding model.

    Args:
        text (str): The text to convert to embeddings.

    Returns:
        numpy.ndarray: The generated embeddings.
    """
    try:
        embeddings_input = embedding_tokenizer.encode(text, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            embeddings = (
                embedding_model.encoder(embeddings_input)
                .last_hidden_state.mean(dim=1)
                .cpu()
                .numpy()
            )
        return embeddings
    except Exception as e:
        logger.error(f"Error during text to embeddings conversion: {e}")
        raise


# Create two columns for the chat interface
col1, col2 = st.columns([2, 1])

# Retrieve conversation history from the database
try:
    conversation_history = conn.execute(
        "SELECT input, output FROM embeddings"
    ).fetchall()
except Exception as e:
    logger.error(f"Error retrieving conversation history: {e}")
    st.error("Failed to retrieve conversation history. Please check the logs.")

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
            try:
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
                logger.info("Conversation embeddings stored.")
            except Exception as e:
                logger.error(f"Error during embedding storage: {e}")
                st.error(
                    "Failed to store conversation embeddings. Please check the logs."
                )

            # Update the conversation history in the Streamlit app
            conversation_container.empty()
            with conversation_container.container():
                try:
                    for input_text, output_text in conn.execute(
                        "SELECT input, output FROM embeddings"
                    ).fetchall():
                        st.markdown(f"**User:** {input_text}")
                        st.markdown(f"**Assistant:** {output_text}")
                        st.markdown("---")
                except Exception as e:
                    logger.error(f"Error updating conversation history: {e}")
                    st.error(
                        "Failed to update conversation history. Please check the logs."
                    )

# Instructions to run the Streamlit app on a local machine
# streamlit run /home/lloyd/Downloads/PythonScripts/localphi3llmtextgeneration.py --server.port 8501 --server.address 0.0.0.0
