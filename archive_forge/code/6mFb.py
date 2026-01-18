import streamlit as st
import openai
from dotenv import load_dotenv
import os
from cryptography.fernet import Fernet
from openai.types import Model, Completion
from openai import OpenAI
import json
import sqlite3
import bcrypt

# Load environment variables
load_dotenv()


# Manage security for API keys
class SecurityManager:
    def __init__(self):
        crypto_key = os.getenv("CRYPTO_KEY")
        if not crypto_key:
            if st.sidebar.button("Create CRYPTO_KEY"):
                generated_key = Fernet.generate_key().decode()
                os.environ["CRYPTO_KEY"] = generated_key
                with open(".env", "a") as env_file:
                    env_file.write(f"\nCRYPTO_KEY={generated_key}")
                st.sidebar.success("CRYPTO_KEY created and saved.")
            else:
                st.sidebar.error(
                    "CRYPTO_KEY environment variable not set. Please create one."
                )
                raise ValueError("CRYPTO_KEY environment variable not set.")
        self.cipher_suite = Fernet(crypto_key)

    def encrypt(self, data: str) -> str:
        """Encrypts the provided data using Fernet encryption."""
        return self.cipher_suite.encrypt(data.encode()).decode()

    def decrypt(self, data: str) -> str:
        """Decrypts the provided data using Fernet decryption."""
        try:
            return self.cipher_suite.decrypt(data.encode()).decode()
        except Fernet.InvalidToken:
            raise ValueError(
                "Decryption failed. Check if the encrypted data or key has changed."
            )


# Configuration management
class ConfigManager:
    def __init__(self, security_manager=None, user_id=None):
        self.user_id = user_id
        if security_manager is None:
            security_manager = SecurityManager()
        self.security_manager = security_manager
        self.settings = self.load_settings()

    def update_settings(self, new_settings: dict):
        """Updates the configuration settings."""
        if self.validate_settings(new_settings):
            self.settings.update(new_settings)
            self.save_settings()
        else:
            st.error("Invalid settings. Please check the values.")

    def validate_settings(self, settings: dict) -> bool:
        """Validates settings to ensure they meet criteria."""
        if settings.get("max_tokens") and not (1 <= settings["max_tokens"] <= 500):
            return False
        return True

    def load_settings(self) -> dict:
        """Loads settings from a user-specific file if available."""
        if self.user_id:
            try:
                with open(f"{self.user_id}_settings.json", "r") as file:
                    return json.load(file)
            except FileNotFoundError:
                return self.load_global_settings()
        return self.load_global_settings()

    def save_settings(self):
        """Saves settings to a user-specific file."""
        if self.user_id:
            with open(f"{self.user_id}_settings.json", "w") as file:
                json.dump(self.settings, file)
        else:
            self.save_global_settings()

    def load_global_settings(self) -> dict:
        """Loads global settings from environment variables."""
        settings = {
            "api_key_encrypted": os.getenv("ENCRYPTED_API_KEY", ""),
            "model": os.getenv("MODEL", "text-davinci-002"),
            "max_tokens": int(os.getenv("MAX_TOKENS", 150)),
            "temperature": float(os.getenv("TEMPERATURE", 0.7)),
            "top_p": float(os.getenv("TOP_P", 1.0)),
            "frequency_penalty": float(os.getenv("FREQUENCY_PENALTY", 0.0)),
            "presence_penalty": float(os.getenv("PRESENCE_PENALTY", 0.0)),
        }
        return settings

    def save_global_settings(self):
        """Saves settings to a persistent store."""
        for key, value in self.settings.items():
            os.environ[key.upper()] = str(value)
            with open(".env", "a") as env_file:
                env_file.write(f"\n{key.upper()}={value}")

    def encrypt_api_key(self, key: str) -> str:
        """Encrypts the API key."""
        return self.security_manager.encrypt(key)

    def decrypt_api_key(self) -> str:
        """Decrypts the API key."""
        encrypted_key = self.settings.get("api_key_encrypted", "")
        if not encrypted_key:
            st.warning("API key is not set. Please configure it in the settings.")
            return None  # Return None to handle this gracefully
        return self.security_manager.decrypt(encrypted_key)


# OpenAI API interaction
class OpenAIInterface:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def get_response(self, prompt: str, model_params: dict) -> str:
        """Fetches response for the given prompt using the specified model parameters."""
        try:
            response = self.client.completions.create(
                model=model_params["model"],
                prompt=prompt,
                max_tokens=model_params["max_tokens"],
                temperature=model_params["temperature"],
                top_p=model_params["top_p"],
                frequency_penalty=model_params["frequency_penalty"],
                presence_penalty=model_params["presence_penalty"],
            )
            return response.choices[0].text.strip()
        except Exception as e:
            st.error("Failed to get response: " + str(e))
            return ""

    def fetch_model_options(self) -> list:
        """Fetches available models from OpenAI."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            st.error("Failed to fetch models: " + str(e))
            return []

    def authenticate(
        self,
        username: str,
        password: str,
        email: str = None,
        verification_code: str = None,
    ) -> bool:
        """Authenticates a user or registers a new one if email and verification code are provided."""
        user_data = self._load_user_data(username)
        if user_data:
            return self._verify_password(password, user_data["password_hash"])
        elif email and verification_code:
            if self._verify_verification_code(email, verification_code):
                password_hash = self._hash_password(password)
                self._save_user_data(username, password_hash, email)
                return True
            else:
                st.error("Invalid verification code.")
                return False
        else:
            st.error(
                "User not found. Please provide an email and verification code to register."
            )
            return False

    def _load_user_data(self, username: str) -> dict:
        """Loads user data from a persistent store."""
        # This should interact with your user database or storage system
        return {}

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verifies the password against the stored hash."""
        return bcrypt.checkpw(password.encode(), password_hash.encode())

    def _hash_password(self, password: str) -> str:
        """Hashes the password using bcrypt."""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def _save_user_data(self, username: str, password_hash: str, email: str):
        """Saves the user data."""
        # This should save to your user database or storage system
        pass

    def _verify_verification_code(self, email: str, code: str) -> bool:
        """Verifies the provided verification code against the stored one for the email."""
        # This should check the code against your database or storage system
        return True


# Database handler
class DatabaseHandler:
    def __init__(self, db_path="embeddings.db"):
        self.conn = sqlite3.connect(db_path)
        self.ensure_tables()

    def ensure_tables(self):
        """Ensures all necessary tables are created."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB NOT NULL
            );
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model BLOB NOT NULL
            );
        """
        )
        self.conn.commit()

    def insert_embedding(self, embedding):
        """Inserts an embedding into the database."""
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO embeddings (embedding) VALUES (?)", (embedding,))
        self.conn.commit()

    def fetch_embedding(self, embedding_id):
        """Fetches an embedding by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT embedding FROM embeddings WHERE id = ?", (embedding_id,))
        result = cursor.fetchone()
        return result[0] if result else None


# Streamlit application
def app():
    if "authenticated" not in st.session_state:
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if authenticate(
                username, password
            ):  # Implement this function based on your auth system
                st.session_state["authenticated"] = True
            else:
                st.error("Authentication failed.")

    if st.session_state.get("authenticated", False):
        st.title("Indego Chat Assistant")
        config_manager = ConfigManager(user_id=username)
        if "config" not in st.session_state:
            st.session_state["config"] = config_manager.settings

        # Fetch and update model options before configuring the sidebar
        api_interface = OpenAIInterface(config_manager.decrypt_api_key())
        model_options = api_interface.fetch_model_options()
        if config_manager.settings["model"] not in model_options:
            config_manager.update_settings({"model": model_options[0]})

        # Initialize the interface without loading a model
        if "model_initialized" not in st.session_state:
            st.session_state["model_initialized"] = False

        if not st.session_state["model_initialized"]:
            model_choice = st.selectbox("Choose a model", options=model_options)
            if st.button("Initialize Model"):
                config_manager.update_settings({"model": model_choice})
                st.session_state["model_initialized"] = True
            return  # Stop further execution until a model is initialized

        configure_settings_sidebar(config_manager)

        user_input, api_interface = initialize_chat(config_manager)

        if st.button("Send"):
            process_user_input(user_input, api_interface)


def configure_settings_sidebar(config_manager):
    with st.sidebar:
        st.title("Configuration")
        api_key = st.text_input(
            "API Key", type="password", value=config_manager.decrypt_api_key() or ""
        )
        if st.button("Encrypt and Store API Key"):
            encrypted_key = config_manager.encrypt_api_key(api_key)
            config_manager.update_settings({"api_key_encrypted": encrypted_key})
            st.success("API Key encrypted and stored.")

        api_interface = OpenAIInterface(config_manager.decrypt_api_key())
        model_options = api_interface.fetch_model_options()
        model = st.selectbox(
            "Model",
            options=model_options,
            index=model_options.index(config_manager.settings["model"]),
        )
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=500,
            value=config_manager.settings["max_tokens"],
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=config_manager.settings["temperature"],
        )
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=config_manager.settings["top_p"],
        )
        frequency_penalty = st.slider(
            "Frequency Penalty",
            min_value=0.0,
            max_value=2.0,
            value=config_manager.settings["frequency_penalty"],
        )
        presence_penalty = st.slider(
            "Presence Penalty",
            min_value=0.0,
            max_value=2.0,
            value=config_manager.settings["presence_penalty"],
        )

        config_manager.update_settings(
            {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }
        )


def initialize_chat(config_manager):
    """Handles the initial setup for user interaction."""
    user_input = st.text_area("Type your message here:")
    api_key = config_manager.decrypt_api_key()
    api_interface = OpenAIInterface(api_key)
    return user_input, api_interface


def process_user_input(user_input, api_interface):
    """Process and display response for user input."""
    if user_input:
        model_params = st.session_state["config"]
        ai_response = api_interface.get_response(user_input, model_params)
        st.write(f"Indego: {ai_response}")


def authenticate(username, password):
    """Authenticates the user."""
    # Implement your authentication logic here
    # This is a placeholder function
    return True


# Ensure correct usage by running through Streamlit
if __name__ == "__main__":
    load_dotenv()  # Ensure environment variables are loaded
    security_manager = SecurityManager()  # Instantiate SecurityManager
    config_manager = ConfigManager(
        security_manager=security_manager
    )  # Pass it to ConfigManager
    app()
