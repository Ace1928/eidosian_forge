import streamlit as st
import openai
from dotenv import load_dotenv
import os
from cryptography.fernet import Fernet

# Load environment variables
load_dotenv()


# Manage security for API keys
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
        pass

    def encrypt_api_key(self, key):
        return self.security_manager.encrypt(key)

    def decrypt_api_key(self):
        encrypted_key = self.settings["api_key_encrypted"]
        if not encrypted_key:
            raise ValueError("API key is not set.")
        return self.security_manager.decrypt(encrypted_key)


# OpenAI API interaction
class OpenAIInterface:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def get_text_response(self, prompt, model_params):
        response = openai.Completion.create(
            engine=model_params["model"],
            prompt=prompt,
            max_tokens=model_params["max_tokens"],
            temperature=model_params["temperature"],
            top_p=model_params["top_p"],
            frequency_penalty=model_params["frequency_penalty"],
            presence_penalty=model_params["presence_penalty"],
        )
        return response.choices[0].text.strip()

    def generate_image(self, prompt):
        response = openai.Image.create(
            prompt=prompt, n=1  # Number of images to generate
        )
        return response.data[0].url

    def transcribe_audio(self, audio_file_path):
        response = openai.Audio.transcriptions.create(
            file=openai.File.upload(file_path=audio_file_path)
        )
        return response.transcription_text

    def fetch_model_options(self):
        models = openai.Engine.list()
        return [model.id for model in models.data]


# Streamlit application setup
def app():
    st.title("Advanced OpenAI Services Interface")
    security_manager = SecurityManager()
    config_manager = ConfigManager(security_manager)
    api_interface = OpenAIInterface(config_manager.decrypt_api_key())

    # UI for different services
    service_type = st.selectbox(
        "Select Service", ["Text Completion", "Image Generation", "Audio Transcription"]
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
