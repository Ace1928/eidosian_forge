import streamlit as st
import openai
from dotenv import load_dotenv
import os
import json
from cryptography.fernet import Fernet
from openai.types import (
    Completion,
    CreateEmbeddingResponse,
    FileObject,
    ImagesResponse,
    Model,
    Batch,
)

# Load environment variables
load_dotenv()


# Security management for API keys
class SecurityManager:
    def __init__(self):
        crypto_key = os.getenv("CRYPTO_KEY")
        if not crypto_key:
            raise ValueError("CRYPTO_KEY environment variable not set.")
        self.cipher_suite = Fernet(crypto_key)

    def encrypt(self, data):
        return self.cipher_suite.encrypt(data.encode()).decode()

    def decrypt(self, data):
        try:
            return self.cipher_suite.decrypt(data.encode()).decode()
        except Fernet.InvalidToken:
            raise ValueError(
                "Decryption failed. Check if the encrypted data or key has changed."
            )


# Configuration management
class ConfigManager:
    def __init__(self, security_manager):
        self.security_manager = security_manager
        self.settings = self.load_settings()

    def update_settings(self, new_settings):
        self.settings.update(new_settings)
        self.save_settings()

    def load_settings(self):
        # Load settings from a persistent source if available
        return {
            "api_key_encrypted": os.getenv("ENCRYPTED_API_KEY", ""),
            "model": "text-davinci-002",
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

    def save_settings(self):
        # Implement saving to a persistent store
        pass

    def encrypt_api_key(self, key):
        return self.security_manager.encrypt(key)

    def decrypt_api_key(self):
        encrypted_key = self.settings["api_key_encrypted"]
        if not encrypted_key:
            raise ValueError("API key is not set.")
        return self.security_manager.decrypt(encrypted_key)


# OpenAI API interaction for multiple services
class OpenAIInterface:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_text_response(self, prompt, model_params) -> Completion:
        openai.api_key = self.api_key
        return openai.Completion.create(
            engine=model_params["model"],
            prompt=prompt,
            max_tokens=model_params["max_tokens"],
            temperature=model_params["temperature"],
            top_p=model_params["top_p"],
            frequency_penalty=model_params["frequency_penalty"],
            presence_penalty=model_params["presence_penalty"],
        )

    def create_embedding(self, input, model) -> CreateEmbeddingResponse:
        openai.api_key = self.api_key
        return openai.Embedding.create(input=input, model=model)

    def upload_file(self, file, purpose) -> FileObject:
        openai.api_key = self.api_key
        return openai.File.create(file=file, purpose=purpose)

    def generate_image(self, prompt) -> ImagesResponse:
        openai.api_key = self.api_key
        return openai.Image.create(prompt=prompt, n=1)

    def fetch_model_options(self) -> list[Model]:
        openai.api_key = self.api_key
        return openai.Model.list()

    def create_fine_tune(
        self, training_file, model, n_epochs, batch_size
    ) -> FineTuningJob:
        openai.api_key = self.api_key
        return openai.FineTune.create(
            training_file=training_file,
            model=model,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )

    def create_vector_store(self, name, description) -> VectorStore:
        openai.api_key = self.api_key
        return openai.VectorStore.create(name=name, description=description)

    def upload_vector_store_file(self, vector_store_id, file) -> VectorStoreFile:
        openai.api_key = self.api_key
        return openai.VectorStoreFile.create(vector_store_id=vector_store_id, file=file)

    def create_vector_store_file_batch(
        self, vector_store_id, files
    ) -> VectorStoreFileBatch:
        openai.api_key = self.api_key
        return openai.VectorStoreFileBatch.create(
            vector_store_id=vector_store_id, files=files
        )

    def create_assistant(self, name, description) -> Assistant:
        openai.api_key = self.api_key
        return openai.Assistant.create(name=name, description=description)

    def create_thread(self, assistant_id, name, prompt) -> Thread:
        openai.api_key = self.api_key
        return openai.Thread.create(assistant_id=assistant_id, name=name, prompt=prompt)

    def create_run(self, thread_id) -> Run:
        openai.api_key = self.api_key
        return openai.Run.create(thread_id=thread_id)

    def retrieve_run_step(self, thread_id, run_id, step_id) -> RunStep:
        openai.api_key = self.api_key
        return openai.RunStep.retrieve(
            thread_id=thread_id, run_id=run_id, step_id=step_id
        )

    def create_message(self, thread_id, content) -> Message:
        openai.api_key = self.api_key
        return openai.Message.create(thread_id=thread_id, content=content)

    def create_batch(self, requests) -> Batch:
        openai.api_key = self.api_key
        return openai.Batch.create(requests=requests)


# Streamlit application setup
def app():
    st.title("Advanced OpenAI Services Interface")
    security_manager = SecurityManager()
    config_manager = ConfigManager(security_manager)
    api_key = config_manager.decrypt_api_key()
    api_interface = OpenAIInterface(api_key)

    # UI for different services
    service_type = st.sidebar.selectbox(
        "Select Service",
        [
            "Text Completion",
            "Chat Completion",
            "Embeddings",
            "File Upload",
            "Image Generation",
            "Audio Transcription",
            "Audio Translation",
            "Content Moderation",
            "Models",
            "Fine-tuning",
            "Vector Stores",
            "Assistants",
            "Threads",
            "Runs",
            "Messages",
            "Batches",
        ],
    )

    if service_type == "Text Completion":
        user_input = st.text_area("Type your message here:")
        if st.button("Send"):
            model_params = config_manager.settings
            response = api_interface.get_text_response(user_input, model_params)
            st.write(f"Response: {response.choices[0].text.strip()}")

    elif service_type == "Chat Completion":
        messages = st.session_state.get("messages", [])
        user_input = st.text_input("Type your message here:")
        if st.button("Send"):
            messages.append({"role": "user", "content": user_input})
            model_params = config_manager.settings
            response = api_interface.get_chat_response(messages, model_params)
            messages.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )
            st.session_state["messages"] = messages
        for message in messages:
            st.write(f"{message['role'].capitalize()}: {message['content']}")

    elif service_type == "Embeddings":
        input_text = st.text_area("Enter text to generate embeddings:")
        model = st.selectbox(
            "Select model", ["text-embedding-ada-002", "text-search-ada-doc-001"]
        )
        if st.button("Generate Embeddings"):
            response = api_interface.create_embedding(input_text, model)
            st.write(f"Embeddings: {response.data[0].embedding}")

    elif service_type == "File Upload":
        uploaded_file = st.file_uploader("Choose a file")
        purpose = st.text_input("Enter purpose of the file")
        if st.button("Upload File") and uploaded_file is not None:
            response = api_interface.upload_file(uploaded_file, purpose)
            st.write(f"File uploaded. ID: {response.id}")

    elif service_type == "Image Generation":
        prompt = st.text_input("Image Prompt")
        if st.button("Generate"):
            response = api_interface.generate_image(prompt)
            st.image(response.data[0].url, caption="Generated Image")

    elif service_type == "Audio Transcription":
        audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])
        if st.button("Transcribe") and audio_file:
            response = api_interface.transcribe_audio(audio_file)
            st.write(f"Transcription: {response.text}")

    elif service_type == "Audio Translation":
        audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])
        if st.button("Translate") and audio_file:
            response = api_interface.translate_audio(audio_file)
            st.write(f"Translation: {response.text}")

    elif service_type == "Content Moderation":
        input_text = st.text_area("Enter text to moderate:")
        if st.button("Moderate"):
            response = api_interface.moderate_content(input_text)
            st.write(f"Moderation Result: {response.results[0]}")

    elif service_type == "Models":
        models = api_interface.fetch_model_options()
        st.write("Available Models:")
        for model in models.data:
            st.write(f"- {model.id}")

    elif service_type == "Fine-tuning":
        training_file_id = st.text_input("Enter training file ID")
        model = st.text_input("Enter base model name")
        n_epochs = st.number_input("Number of epochs", min_value=1, value=4)
        batch_size = st.number_input("Batch size", min_value=1, value=1)
        if st.button("Create Fine-tune"):
            response = api_interface.create_fine_tune(
                training_file_id, model, n_epochs, batch_size
            )
            st.write(f"Fine-tuning job created. ID: {response.id}")

    elif service_type == "Vector Stores":
        name = st.text_input("Vector Store Name")
        description = st.text_area("Vector Store Description")
        if st.button("Create Vector Store"):
            response = api_interface.create_vector_store(name, description)
            st.write(f"Vector Store created. ID: {response.id}")

        vector_store_id = st.text_input("Vector Store ID")
        vector_store_file = st.file_uploader("Upload file to Vector Store")
        if st.button("Upload File to Vector Store") and vector_store_file is not None:
            response = api_interface.upload_vector_store_file(
                vector_store_id, vector_store_file
            )
            st.write(f"File uploaded to Vector Store. ID: {response.id}")

        vector_store_files = st.file_uploader(
            "Upload files to Vector Store", accept_multiple_files=True
        )
        if st.button("Create File Batch in Vector Store") and vector_store_files:
            response = api_interface.create_vector_store_file_batch(
                vector_store_id, vector_store_files
            )
            st.write(f"File Batch created in Vector Store. ID: {response.id}")

    elif service_type == "Assistants":
        name = st.text_input("Assistant Name")
        description = st.text_area("Assistant Description")
        if st.button("Create Assistant"):
            response = api_interface.create_assistant(name, description)
            st.write(f"Assistant created. ID: {response.id}")

    elif service_type == "Threads":
        assistant_id = st.text_input("Assistant ID")
        name = st.text_input("Thread Name")
        prompt = st.text_area("Thread Prompt")
        if st.button("Create Thread"):
            response = api_interface.create_thread(assistant_id, name, prompt)
            st.write(f"Thread created. ID: {response.id}")

    elif service_type == "Runs":
        thread_id = st.text_input("Thread ID")
        if st.button("Create Run"):
            response = api_interface.create_run(thread_id)
            st.write(f"Run created. ID: {response.id}")

    elif service_type == "Messages":
        thread_id = st.text_input("Thread ID")
        content = st.text_area("Message Content")
        if st.button("Create Message"):
            response = api_interface.create_message(thread_id, content)
            st.write(f"Message created. ID: {response.id}")

    elif service_type == "Batches":
        requests = st.text_area("Enter batch requests (JSON)")
        if st.button("Create Batch"):
            try:
                parsed_requests = json.loads(requests)
                response = api_interface.create_batch(parsed_requests)
                st.write(f"Batch created. ID: {response.id}")
            except json.JSONDecodeError:
                st.error("Invalid JSON format for batch requests.")


if __name__ == "__main__":
    app()
