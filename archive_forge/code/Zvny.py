import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from cryptography.fernet import Fernet

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Manage security for API keys
class SecurityManager:
    def __init__(self):
        crypto_key = os.getenv("CRYPTO_KEY")
        if not crypto_key:
            raise ValueError("CRYPTO_KEY environment variable not set.")
        self.cipher_suite = Fernet(crypto_key)

    def encrypt(self, data: str) -> str:
        return self.cipher_suite.encrypt(data.encode()).decode()

    def decrypt(self, data: str) -> str:
        try:
            return self.cipher_suite.decrypt(data.encode()).decode()
        except Fernet.InvalidToken:
            raise ValueError(
                "Decryption failed. Check if the encrypted data or key has changed."
            )


# Configuration management
class ConfigManager:
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.settings = {}
        self.OpenAIInterface = OpenAIInterface(api_key="")

    def update_settings(self, new_settings: dict):
        """Update the current settings with new settings and save them."""
        self.settings.update(new_settings)
        self.save_settings()

    def load_settings(self) -> dict:
        """
        Loads settings from a JSON file. If the file is not found or an error occurs,
        returns the default settings.
        """
        try:
            with open("indego_chat_settings.json", "r") as file:
                self.settings = json.load(file)
        except FileNotFoundError:
            self.settings = self.get_default_settings()
            self.save_settings()  # Save the default settings if file not found
        except Exception as e:
            st.error(f"Failed to load settings: {str(e)}")
            self.settings = self.get_default_settings()
        return self.settings

    def get_default_settings(self) -> dict:
        """Returns default model settings."""
        return {
            "model": "gpt-3.5-turbo",
            "max_tokens": 2048,
            "temperature": 0.95,
            "top_p": 1.0,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.05,
        }

    def save_settings(self):
        """Saves current settings to a JSON file. Creates the file if it does not exist."""
        try:
            with open("indego_chat_settings.json", "w") as file:
                json.dump(self.settings, file)
        except Exception as e:
            st.error(f"Failed to save settings: {str(e)}")

    def fetch_model_options(self):
        """Fetches and returns available model options from OpenAI."""
        try:
            models = (
                self.OpenAIInterface.fetch_model_options()
            )  # Assuming this method exists in OpenAIInterface
            model_options = [model.id for model in models]
            return model_options
        except Exception as e:
            st.error(f"Failed to fetch model options: {str(e)}")
            return []


# OpenAI API interaction
class OpenAIInterface:
    def __init__(self, api_key: str = os.getenv("OPENAI_API_KEY")):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def get_text_response(
        self, prompt: str, model: str = "text-davinci-002", **params
    ) -> str:
        try:
            response = client.Completion.create(model=model, prompt=prompt, **params)
            return response.get("choices", [{}])[0].get("text", "").strip()
        except Exception as e:
            print(f"Error in get_text_response: {str(e)}")
            return "Error processing text completion."

    def generate_image(self, prompt: str, model: str = "dall-e-2", **params) -> str:
        try:
            response = client.Image.create(prompt=prompt, model=model, **params)
            return response.data[0].url
        except Exception as e:
            print(f"Error in generate_image: {str(e)}")
            return "Error generating image."

    def transcribe_audio(self, file_id: str) -> str:
        try:
            response = client.audio.transcriptions.create(file=file_id)
            return response.transcription_text
        except Exception as e:
            print(f"Error in transcribe_audio: {str(e)}")
            return "Error transcribing audio."

    def fetch_model_options(self) -> list:
        try:
            models = client.models.list()
            model_options = [model for model in models]
            return model_options
        except Exception as e:
            print(f"Failed to fetch model options: {str(e)}")
            return []

    def upload_file(self, file_path: str, purpose: str) -> str:
        try:
            response = client.files.create(file=open(file_path, "rb"), purpose=purpose)
            return response.id
        except Exception as e:
            print(f"Error in upload_file: {str(e)}")
            return "Error uploading file."

    def delete_file(self, file_id: str) -> str:
        try:
            response = client.files.delete(file_id)
            return "File deleted successfully."
        except Exception as e:
            print(f"Error in delete_file: {str(e)}")
            return "Error deleting file."

    def create_fine_tuning_job(self, training_file: str, model: str, **params) -> str:
        try:
            response = client.fine_tuning.create(
                training_file=training_file, model=model, **params
            )
            return response.id
        except Exception as e:
            print(f"Error in create_fine_tuning_job: {str(e)}")
            return "Error creating fine-tuning job."

    def create_vector_store(self, name: str, **params) -> str:
        try:
            response = client.vector_stores.create(name=name, **params)
            return response.id
        except Exception as e:
            print(f"Error in create_vector_store: {str(e)}")
            return "Error creating vector store."

    def batch_create(self, items: list, **params) -> str:
        try:
            response = client.batches.create(items=items, **params)
            return response.id
        except Exception as e:
            print(f"Error in batch_create: {str(e)}")
            return "Error processing batch creation."


# Streamlit application setup
def app():
    st.title("Advanced OpenAI Services Interface")
    security_manager = SecurityManager()
    config_manager = ConfigManager(security_manager)

    with st.sidebar:
        st.header("Configuration")
        if st.button("Load Configuration"):
            config_manager.settings = config_manager.load_settings()
        if st.button("Save Configuration"):
            config_manager.save_settings()

        api_key = st.text_input("API Key", type="password")

        model_options = config_manager.fetch_model_options() if api_key else []
        selected_model = st.selectbox(
            "Model",
            options=model_options,
            index=(
                model_options.index(
                    config_manager.settings.get("model", "text-davinci-002")
                )
                if "model" in config_manager.settings
                else 0
            ),
        )
        config_manager.settings["model"] = selected_model

        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=2048,
            value=config_manager.settings.get("max_tokens", 150),
        )
        config_manager.settings["max_tokens"] = max_tokens

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=config_manager.settings.get("temperature", 0.7),
        )
        config_manager.settings["temperature"] = temperature

        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=config_manager.settings.get("top_p", 1.0),
        )
        config_manager.settings["top_p"] = top_p

        frequency_penalty = st.slider(
            "Frequency Penalty",
            min_value=0.0,
            max_value=1.0,
            value=config_manager.settings.get("frequency_penalty", 0.0),
        )
        config_manager.settings["frequency_penalty"] = frequency_penalty

        presence_penalty = st.slider(
            "Presence Penalty",
            min_value=0.0,
            max_value=1.0,
            value=config_manager.settings.get("presence_penalty", 0.0),
        )
        config_manager.settings["presence_penalty"] = presence_penalty

    if st.button("Initialize API Interface"):
        api_interface = OpenAIInterface(config_manager.decrypt_api_key())

        # UI for different services
        service_type = st.selectbox(
            "Select Service",
            ["Text Completion", "Image Generation", "Audio Transcription"],
        )

        if service_type == "Text Completion":
            user_input = st.text_area("Type your message here:")
            if st.button("Send"):
                model_params = config_manager.settings
                response = api_interface.get_text_response(user_input, model_params)
                st.write(f"Response: {response}")
        elif service_type == "Image Generation":
            prompt = st.text_input("Image Prompt")
            if st.button("Generate"):
                image_url = api_interface.generate_image(prompt)
                st.image(image_url, caption="Generated Image")
        elif service_type == "Audio Transcription":
            audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])
            if st.button("Transcribe") and audio_file:
                response_text = api_interface.transcribe_audio(audio_file)
                st.write(f"Transcription: {response_text}")


if __name__ == "__main__":
    app()
