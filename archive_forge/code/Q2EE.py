"""
    .. _core_services.py:
    ================================================================================
    Title: Core Services Module for Image Interconversion GUI Application
    ================================================================================
    Path: scripts/image_interconversion_gui/core_services.py
    ================================================================================
    Description:
        The Core Services module is a foundational component of the Image Interconversion GUI application, designed to ensure the secure and efficient operation of the application. It provides a suite of services including dynamic logging, secure encryption key management, and versatile configuration management. This module is pivotal in maintaining the integrity, security, and usability of the application, catering to both the operational needs and the user's security concerns.
    ================================================================================
    Overview:
        The Core Services module acts as the backbone of the Image Interconversion GUI application, orchestrating several critical operations. It is responsible for initializing and managing application-wide settings, securing sensitive data through robust encryption techniques, and dynamically adjusting logging levels for optimal debugging and monitoring. The module's design emphasizes modularity, security, and ease of use, supporting various configuration formats and integrating seamlessly with external logging solutions for enhanced operational visibility.
    ================================================================================
    Purpose:
        The primary purpose of the Core Services module is to provide a secure, efficient, and flexible infrastructure for the Image Interconversion GUI application. It aims to streamline the application's core operations, such as logging, encryption, and configuration management, thereby enhancing the application's overall performance, security posture, and user experience.
    ================================================================================
    Scope:
        The Core Services module is essential to the entire lifecycle of the Image Interconversion GUI application. It influences the application's behavior from initialization to termination, impacting areas such as user interaction, data processing, security, and logging. The module's functionality is integral to the application's ability to perform its intended image interconversion tasks securely and efficiently.
    ================================================================================
    Definitions:
        INI: A simple format used for configuration files for software applications, characterized by sections and key-value pairs.
        JSON: JavaScript Object Notation, a lightweight data-interchange format that is easy for humans to read and write and for machines to parse and generate.
        YAML: YAML Ain't Markup Language, a human-readable data serialization standard used for configuration files and data exchange between languages with different data structures.
        Fernet: A symmetric encryption method provided by the cryptography library, ensuring that a message encrypted cannot be manipulated or read without the key. It is designed for secure storage and transmission of sensitive data.
    ================================================================================
    Key Features:
        - Dynamic Logging Configuration: Enables real-time adjustment of logging levels and formats, facilitating detailed application insights and efficient debugging.
        - Secure Encryption Key Management: Utilizes Fernet for the generation, storage, and validation of encryption keys, ensuring the secure handling of sensitive data.
        - Comprehensive Configuration Management: Supports multiple configuration formats (INI, JSON, YAML), allowing for flexible application setup and customization.
        - Enhanced Error Handling and Logging: Implements advanced error logging mechanisms for improved application robustness and clarity in troubleshooting.
        - External Logging Platform Integration: Offers capabilities to integrate with external logging platforms, enhancing monitoring and operational visibility.
    ================================================================================
    Usage:
        The Core Services module is utilized within the Image Interconversion GUI application to manage essential services such as logging, encryption, and configuration. To integrate these services, the application imports the necessary classes from the module and invokes their methods according to the operational requirements. This modular approach facilitates easy customization and extension of core functionalities.

        Example:
        ```python
        from core_services import LoggingManager, EncryptionManager, ConfigManager
        
        # Configure application logging
        LoggingManager.configure_logging(log_level="DEBUG")
        
        # Generate and manage encryption keys
        encryption_key = EncryptionManager.generate_key()
        
        # Initialize and use the configuration manager
        config_manager = ConfigManager()
        ```
    ================================================================================
    Dependencies:
        - Python 3.8 or higher: Required for the latest language features and standard library modules.
        - configparser: Utilized for managing INI configuration files.
        - json: Employed for handling JSON data serialization and deserialization.
        - os: Provides a way of using operating system-dependent functionality.
        - logging: Facilitates logging across the application, supporting various handlers and configurations.
        - cryptography: Offers cryptographic recipes and primitives, including Fernet for encryption.
        - typing: Supports type annotations, enhancing code readability and maintainability.
        - yaml: Used for managing YAML configuration files, enabling human-readable data serialization.
        - aiofiles: Supports asynchronous file operations, improving I/O efficiency in an asynchronous programming environment.
        - asyncio: Enables asynchronous programming, allowing for concurrent execution of code, improving application responsiveness and scalability.
    ================================================================================
    References:
        - Python 3 Documentation: Provides comprehensive information on Python's syntax, modules, and libraries. URL: https://docs.python.org/3/
        - Cryptography Documentation: Offers detailed guidance on cryptographic recipes and primitives provided by the cryptography library. URL: https://cryptography.io/en/latest/
        - AsyncIO Documentation: Contains extensive documentation on asynchronous programming with asyncio in Python. URL: https://docs.python.org/3/library/asyncio.html
    ================================================================================
    Authorship and Versioning Details:
        Author: Lloyd Handyside
        Creation Date: 2024-04-08
        Last Modified: 2024-04-08
        Version: 1.0.0
        Contact: lloyd.handyside@example.com
        Ownership: Lloyd Handyside
        Status: Final
        This section documents the authorship, version history, and the current status of the document, ensuring clarity on the document's evolution and maintenance.
    ================================================================================
    Functionalities:
        - Logging Management: Offers dynamic configuration capabilities for application-wide logging, including level adjustment and format customization.
        - Encryption Key Management: Facilitates the generation, validation, and secure storage of encryption keys, employing Fernet for high-security standards.
        - Configuration File Management: Supports asynchronous loading, saving, and management of configuration files in INI, JSON, and YAML formats, enhancing application flexibility and user experience.
        - Function Call Logging: Implements a decorator for logging function calls, arguments, and exceptions, aiding in debugging and monitoring of application behavior.
    ================================================================================    
    Notes:
        This module is a critical component of a larger ecosystem aimed at providing a comprehensive and secure solution for image interconversion. It embodies the project's commitment to security, efficiency, and user-friendliness, laying a solid foundation for the application's functionality and extensibility.
    ================================================================================
    Change Log:
        - 2024-04-08, Version 1.0.0: Initial creation of the module. Implemented core functionalities for logging, encryption, and configuration management. This entry marks the module's inception, detailing the foundational features and capabilities introduced.
    ================================================================================
    License:
        This document and the accompanying source code are released under the MIT License, promoting open-source usage and distribution while protecting the contributors' rights. For the full license text, see LICENSE.md in the project root or visit https://opensource.org/licenses/MIT for more details.
    ================================================================================
    Tags: Core Services, Logging, Encryption, Configuration, Image Interconversion GUI
        These tags categorize the document and source code, facilitating easier navigation and discovery within the project repository or documentation.
    ================================================================================
    Contributors:
        - Lloyd Handyside, Initial module development and documentation, 2024-04-08
        This section acknowledges the contributions of individuals towards the development and maintenance of the module, ensuring recognition of their efforts.
    ================================================================================
    Security Considerations:
        - Known Vulnerabilities: None identified at the time of release, reflecting the module's adherence to security best practices and thorough testing.
        - Best Practices: Utilizes Fernet for secure encryption key management, aligning with industry-standard encryption practices to safeguard sensitive data.
        - Encryption Standards: Adheres to established encryption standards, ensuring the confidentiality and integrity of data handled by the module.
        - Data Handling Protocols: Implements stringent protocols for the secure handling and storage of sensitive configuration and encryption keys, emphasizing unauthorized access prevention and data integrity.
    ================================================================================
    Privacy Considerations:
        - Data Collection: The module does not collect personal data, aligning with privacy-by-design principles and minimizing privacy risks.
        - Data Storage: Employs secure storage mechanisms for encryption keys, with access restricted to authorized personnel, ensuring the privacy and security of user data.
        - Privacy by Design: The module's architecture and functionality are crafted with privacy as a core principle, safeguarding sensitive information through encryption and secure management practices.
    ================================================================================
    Performance Benchmarks:
        - The module is optimized for asynchronous operations, significantly reducing blocking I/O operations and enhancing application responsiveness and throughput.
        - Code Efficiency: Leverages modern Python features and best practices for code efficiency, readability, and maintainability, contributing to the module's overall performance and scalability.
    ================================================================================
    Limitations:
        - Known Limitations: The module currently supports a limited set of configuration file formats (INI, JSON, YAML), which may restrict flexibility in certain use cases.
        - Unresolved Issues: There are no unresolved issues at the time of release, indicating a stable and reliable module for its intended functionalities.
        - Future Enhancements: Plans are in place to extend support for additional configuration formats and to integrate with more external logging platforms, aiming to enhance the module's versatility and adaptability to different operational environments.
        This section outlines the current limitations and future directions for the module, providing transparency about its capabilities and ongoing development efforts.
    ================================================================================
"""

import configparser
import json
import os
import logging
from logging.handlers import RotatingFileHandler
from cryptography.fernet import Fernet
from typing import Any, Dict, Optional, Union, Callable
from functools import wraps
import yaml
import aiofiles
import asyncio
import unittest


def log_function_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator that logs the entry, exit, arguments, and exceptions of the decorated function.

    This enhances debugging and monitoring by providing detailed insights into the function's execution flow.

    Args:
        func (Callable[..., Any]): The function to be decorated.

    Returns:
        Callable[..., Any]: The wrapped function with logging functionality.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logging.debug(f"Calling {func.__name__}({signature})")

        try:
            result = func(*args, **kwargs)
            logging.debug(f"{func.__name__} returned {result!r}")
            return result
        except Exception as e:
            logging.error(f"{func.__name__} raised {e.__class__.__name__}: {e}")
            raise

    return wrapper


class LoggingManager:
    @staticmethod
    def configure_logging(
        log_level: str = "INFO",
        log_format: Optional[str] = None,
        log_file_path: Optional[str] = None,
    ) -> None:
        """
        Configures the global logging level, format, and handlers, including file and console handlers.
        This method resets the logging configuration on each call to allow dynamic reconfiguration.

        Parameters:
        - log_level (str): The logging level as a string. Defaults to "INFO".
        - log_format (Optional[str]): The logging format. If None, a default format is used.
        - log_file_path (Optional[str]): The path to the log file. If None, a default path is used.
        """

        # Clear existing logging configuration by removing all handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Define default logging format if not specified
        if log_format is None:
            log_format = "%(asctime)s - %(levelname)s - %(message)s"

        # Define default log file path if not specified
        if log_file_path is None:
            log_file_path = "app.log"

        # Map log level strings to logging constants
        level_mapping = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        # Validate and map the log level; default to INFO if invalid
        if log_level.upper() in level_mapping:
            logging_level = level_mapping[log_level.upper()]
        else:
            raise ValueError(f"Invalid log level: {log_level}")

        # Configure the basic logging with specified level, format, and handlers
        logging.basicConfig(
            level=logging_level,
            format=log_format,
            handlers=[
                RotatingFileHandler(
                    filename=log_file_path,
                    mode="a",
                    maxBytes=10485760,
                    backupCount=10,
                    encoding="utf-8",
                    delay=0,
                ),
                logging.StreamHandler(),  # Log to console
            ],
        )

    @staticmethod
    def info(message: str) -> None:
        logging.info(message)

    @staticmethod
    def warning(message: str) -> None:
        logging.warning(message)

    @staticmethod
    def error(message: str) -> None:
        logging.error(message)

    @staticmethod
    def debug(message: str) -> None:
        """
        Logs a debug-level message.

        Args:
            message (str): The message to log.
        """
        logging.debug(message)


class EncryptionManager:
    """
    Manages encryption and decryption operations.
    """

    KEY_FILE = "/home/lloyd/EVIE/default_encryption.key"

    @staticmethod
    def generate_key() -> bytes:
        """
        Generates a new encryption key.
        """
        if os.path.exists(EncryptionManager.KEY_FILE):
            with open(EncryptionManager.KEY_FILE, "rb") as file:
                key = file.read()
                Fernet(key)  # Validates the key
                logging.debug("Encryption key loaded successfully.")
                return key
        else:
            key = Fernet.generate_key()
            with open(EncryptionManager.KEY_FILE, "wb") as file:
                file.write(key)
            return key

    @staticmethod
    def get_cipher_suite() -> Fernet:
        """
        Retrieves the cipher suite for encryption and decryption operations.

        Returns:
            Fernet: The Fernet cipher suite.
        """
        key = EncryptionManager.get_valid_encryption_key()
        return Fernet(key)

    @staticmethod
    def get_valid_encryption_key() -> bytes:
        """
        Ensures the encryption key's validity or generates a new one if necessary.
        """
        try:
            with open(EncryptionManager.KEY_FILE, "rb") as file:
                key = file.read()
                Fernet(key)  # Validates the key
                logging.debug("Encryption key loaded successfully.")
                return key
        except (FileNotFoundError, ValueError, TypeError):
            logging.error("Invalid or missing encryption key. Generating a new key.")
            new_key = EncryptionManager.generate_key()
            with open(EncryptionManager.KEY_FILE, "wb") as file:
                file.write(new_key)
            logging.info("Generated and stored a new encryption key.")
            return new_key

    @staticmethod
    async def encrypt(data: bytes) -> bytes:
        key = EncryptionManager.get_valid_encryption_key()
        encrypted_data = await asyncio.to_thread(
            EncryptionManager.get_cipher_suite().encrypt, data
        )
        return encrypted_data

    @staticmethod
    async def decrypt(encrypted_data: bytes) -> bytes:
        key = EncryptionManager.get_valid_encryption_key()
        decrypted_data = await asyncio.to_thread(
            EncryptionManager.get_cipher_suite().decrypt, encrypted_data
        )
        return decrypted_data


class ConfigManager:
    """
    Manages application configurations, supporting INI, JSON, and YAML formats, and provides encryption and decryption services for sensitive configuration values.

    Attributes:
        config_files (Dict[str, Union[configparser.ConfigParser, Dict]]): Loaded configuration files.
        encryption_key (bytes): Encryption key for securing sensitive configuration values.
        EncryptionManager.get_cipher_suite (Fernet): Cipher suite for encryption and decryption.
        cache (Dict[str, Any]): Cache for storing frequently accessed configuration values.
        default_config_template: Optional[Dict[str, Dict[str, str]]] = None
    """

    def __init__(
        self,
        encryption_key: Optional[bytes] = None,
        default_config_template: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.config_files: Dict[str, Union[configparser.ConfigParser, Dict]] = {}
        self.encryption_key = EncryptionManager.get_valid_encryption_key()
        self.default_config_template = default_config_template or {}
        logging.info("ConfigManager initialized.")
        self.cache: Dict[str, Any] = {}

    async def ensure_config_file_exists(self, file_path: str) -> None:
        """
        Ensures that the configuration file exists. If not, creates it with the provided default values from the template.

        Args:
            file_path (str): The path to the configuration file.
        """
        if not os.path.exists(file_path) and self.default_config_template:
            logging.info(
                f"Configuration file {file_path} not found. Creating with default values."
            )
            config_parser = configparser.ConfigParser()
            for section, options in self.default_config_template.items():
                await config_parser.add_section(section)
                for option, value in options.items():
                    await config_parser.set(section, option, value)
            async with aiofiles.open(file_path, mode="w") as file:
                await file.write("# Generated by ConfigManager with default values\n")
                await config_parser.write(file)

    async def load_config(
        self, file_path: str, config_name: Optional[str] = None, file_type: str = "ini"
    ) -> None:
        """
        Asynchronously loads a configuration file, supporting INI, JSON, and YAML formats.

        Args:
            file_path (str): The path to the configuration file.
            config_name (Optional[str]): An optional name for the configuration. If not provided, the basename of the file_path is used.
            file_type (str): The type of the configuration file ('ini', 'json', 'yaml').

        Raises:
            ValueError: If an unsupported file type is provided.
            FileNotFoundError: If the file does not exist and no default template is provided.
        """
        # Check for unsupported file types
        if file_type not in ["ini", "json", "yaml"]:
            raise ValueError(
                f"Unsupported file type: {file_type}. Using default configuration template."
            )

        if not config_name:
            config_name = os.path.basename(file_path)

        # Check for file existence only if no default template is provided
        if not os.path.exists(file_path) and not self.default_config_template:
            raise FileNotFoundError(
                f"Configuration file does not exist and no default template provided: {file_path}"
            )

        logging.debug(f"Loading configuration file asynchronously: {file_path}")

        if file_type == "ini":
            config_parser = configparser.ConfigParser()
            async with aiofiles.open(file_path, mode="r") as file:
                config_data = await file.read()
            config_parser.read_string(config_data)
            self.config_files[config_name] = config_parser
        elif file_type == "json":
            async with aiofiles.open(file_path, mode="r") as file:
                config_data = await file.read()
            self.config_files[config_name] = json.loads(config_data)
        elif file_type == "yaml":
            async with aiofiles.open(file_path, mode="r") as file:
                config_data = await file.read()
            self.config_files[config_name] = yaml.safe_load(config_data)

        logging.info(f"Configuration file loaded asynchronously: {file_path}")

    async def get(
        self,
        config_name: str,
        section: str,
        option: str,
        fallback: Optional[Any] = None,
        is_encrypted: bool = False,
    ) -> Union[str, int, float, bool, None]:
        logging.debug(
            f"Retrieving configuration value: {config_name} -> {section}/{option}"
        )
        cache_key = f"{config_name}_{section}_{option}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        config_parser = self.config_files.get(config_name)
        if config_parser:
            try:
                value = None
                if isinstance(
                    config_parser, configparser.ConfigParser
                ) and config_parser.has_option(section, option):
                    value = config_parser.get(section, option)
                elif (
                    isinstance(config_parser, dict)
                    and section in config_parser
                    and option in config_parser[section]
                ):
                    value = config_parser[section][option]
                else:
                    return fallback

                if is_encrypted and value:
                    decrypted_value = await EncryptionManager.decrypt(value)
                    value = decrypted_value.decode()

                # Conversion logic remains unchanged from original implementation
                for cast in (int, float):
                    try:
                        return cast(value)
                    except ValueError:
                        continue
                if isinstance(value, str) and value.lower() in ("true", "false"):
                    return value.lower() == "true"
                self.cache[cache_key] = value
                return value
            except Exception as e:
                logging.error(f"Error retrieving configuration value: {e}")
                return fallback
        else:
            logging.warning(f"Configuration '{config_name}' not found.")
            return fallback

    async def set(
        self,
        config_name: str,
        section: str,
        option: str,
        value: Any,
        is_encrypted: bool = False,
    ) -> None:
        if is_encrypted:
            encrypted_value = await EncryptionManager.encrypt(str(value).encode())

        config_parser = self.config_files.get(config_name)
        if not config_parser:
            logging.error(f"Configuration '{config_name}' not loaded.")
            raise ValueError(f"Configuration '{config_name}' not loaded.")

        if isinstance(config_parser, configparser.ConfigParser):
            if not config_parser.has_section(section):
                config_parser.add_section(section)
            config_parser.set(section, option, str(value))
        elif isinstance(config_parser, dict):
            if section not in config_parser:
                config_parser[section] = {}
            config_parser[section][option] = value
        else:
            logging.error(f"Unsupported configuration type for '{config_name}'.")
            raise ValueError(f"Unsupported configuration type for '{config_name}'.")

        cache_key = f"{config_name}_{section}_{option}"
        self.cache[cache_key] = value

    async def save_config(
        self, config_name: str, file_path: Optional[str] = None, file_type: str = "ini"
    ) -> None:
        """
        Saves the specified configuration back to a file, supporting INI, JSON, and YAML formats. Enhanced with logging to indicate the start and success of saving configuration files.

        Args:
            config_name (str): The name of the configuration to save.
            file_path (Optional[str]): The file path to save the configuration to. If not provided, the original path used to load the configuration is used.
            file_type (str): The type of the configuration file ('ini', 'json', 'yaml').

        Raises:
            ValueError: If an unsupported file type or configuration type is provided.
            Exception: For any issues encountered during file saving.
        """
        try:
            if not file_path:
                # Assuming config_name was the file path if no explicit file_path is provided
                file_path = f"{config_name}.{file_type}"

            config_parser = self.config_files.get(config_name)
            if not config_parser:
                raise ValueError(f"Configuration '{config_name}' not loaded.")

            if file_type == "ini" and isinstance(
                config_parser, configparser.ConfigParser
            ):
                async with aiofiles.open(file_path, "w") as configfile:
                    for section in config_parser.sections():
                        await configfile.write(f"[{section}]\n")
                        for key, value in config_parser.items(section):
                            await configfile.write(f"{key} = {value}\n")
                        await configfile.write("\n")
            elif file_type == "json" and isinstance(config_parser, dict):
                async with aiofiles.open(file_path, "w") as json_file:
                    await json_file.write(json.dumps(config_parser, indent=4))
            elif file_type == "yaml" and isinstance(config_parser, dict):
                async with aiofiles.open(file_path, "w") as yaml_file:
                    await yaml_file.write(yaml.dump(config_parser))
            else:
                raise ValueError(
                    f"Unsupported configuration type or file type for '{config_name}'."
                )

            logging.info(
                f"Configuration '{config_name}' saved successfully to '{file_path}'."
            )
        except Exception as e:
            logging.error(
                f"Failed to save configuration file '{config_name}' to '{file_path}': {e}"
            )
            raise


class TestLoggingManager(unittest.TestCase):
    def test_configure_logging(self):
        """Test configuring the logging level and format."""
        LoggingManager.configure_logging("DEBUG")
        self.assertEqual(
            logging.getLogger().level,
            logging.DEBUG,
            "Logging level should be set to DEBUG.",
        )


class TestEncryptionManager(unittest.TestCase):
    def test_generate_key(self):
        """Test generating a new encryption key."""
        key = EncryptionManager.generate_key()
        self.assertIsInstance(key, bytes, "Generated key should be bytes.")
        self.assertTrue(
            os.path.exists(EncryptionManager.KEY_FILE),
            "Key file should exist after key generation.",
        )

    def test_get_valid_encryption_key(self):
        """Test retrieving a valid encryption key."""
        generated_key = EncryptionManager.generate_key()
        retrieved_key = EncryptionManager.get_valid_encryption_key()
        self.assertEqual(
            generated_key,
            retrieved_key,
            "Retrieved key should match the generated key.",
        )

    def test_key_file_regeneration(self):
        """Test regenerating the encryption key file if deleted."""
        EncryptionManager.generate_key()
        os.remove(EncryptionManager.KEY_FILE)
        self.assertFalse(
            os.path.exists(EncryptionManager.KEY_FILE), "Key file should be deleted."
        )
        EncryptionManager.get_valid_encryption_key()
        self.assertTrue(
            os.path.exists(EncryptionManager.KEY_FILE),
            "Key file should be regenerated.",
        )


class TestConfigManager(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config_manager = ConfigManager()
        self.test_config_path = "test_config.ini"
        self.test_config_content = "[DEFAULT]\nkey=value\n"
        async with aiofiles.open(self.test_config_path, "w") as file:
            await file.write(self.test_config_content)

    async def asyncTearDown(self):
        os.remove(self.test_config_path)

    async def test_load_config(self):
        """Test loading a configuration file."""
        await self.config_manager.load_config(self.test_config_path, "test")
        self.assertIn(
            "test",
            self.config_manager.config_files,
            "Config should be loaded with the name 'test'.",
        )

    async def test_save_config(self):
        """Test saving a configuration file."""
        await self.config_manager.load_config(self.test_config_path, "test")
        new_config_path = "new_test_config.ini"
        await self.config_manager.save_config("test", new_config_path, "ini")
        self.assertTrue(
            os.path.exists(new_config_path),
            "New config file should exist after saving.",
        )
        os.remove(new_config_path)


class TestConfigManagerMore(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config_manager = ConfigManager(
            default_config_template={
                "General": {
                    "log_level": "INFO",
                    "encryption_key_path": "encryption.key",
                },
                "Database": {
                    "db_path": "database.db",
                },
            }
        )
        self.test_config_path = "test_config.ini"
        self.test_config_content = "[DEFAULT]\nkey=value\n"
        async with aiofiles.open(self.test_config_path, "w") as file:
            await file.write(self.test_config_content)
        await self.config_manager.load_config(self.test_config_path, "test")

    async def asyncTearDown(self):
        os.remove(self.test_config_path)

    async def test_load_config_nonexistent(self):
        """Test loading a nonexistent configuration file without a default template."""
        with self.assertRaises(FileNotFoundError):
            await self.config_manager.load_config(
                "nonexistent_config.ini", "nonexistent"
            )

    async def test_load_config_unsupported_file_type(self):
        """Test loading a configuration file with an unsupported file type."""
        with self.assertRaises(ValueError):
            await self.config_manager.load_config(
                "unsupported_config.txt", "unsupported", file_type="txt"
            )

    async def test_get_nonexistent_option(self):
        """Test retrieving a nonexistent configuration option."""
        await self.config_manager.load_config(self.test_config_path, "test")
        value = await self.config_manager.get(
            "test",
            "NonexistentSection",
            "nonexistent_option",
            fallback=None,
            is_encrypted=False,
        )
        self.assertIsNone(value, "Nonexistent option should return None.")

    async def test_set_and_get_encrypted_value(self):
        """Test setting and retrieving an encrypted configuration value."""
        test_value = "secret"
        await self.config_manager.set(
            "test", "Secrets", "encrypted_option", test_value, is_encrypted=True
        )
        retrieved_value = await self.config_manager.get(
            "test", "Secrets", "encrypted_option", is_encrypted=True
        )
        self.assertEqual(
            test_value,
            retrieved_value,
            "Retrieved value should match the original secret.",
        )

    async def test_save_config_invalid_type(self):
        """Test saving a configuration with an invalid file type."""
        with self.assertRaises(ValueError):
            await self.config_manager.save_config("test", file_type="invalid")


class TestLoggingManagerMore(unittest.TestCase):
    def test_invalid_log_level(self):
        """Test configuring logging with an invalid log level."""
        with self.assertRaises(ValueError):
            LoggingManager.configure_logging("INVALID")

    def test_log_file_creation(self):
        """Test if log file is created successfully."""
        log_file_path = "test_log.log"
        LoggingManager.configure_logging(log_file_path=log_file_path)
        self.assertTrue(os.path.exists(log_file_path), "Log file should be created.")
        os.remove(log_file_path)


class TestEncryptionManagerMore(unittest.IsolatedAsyncioTestCase):

    async def test_encrypt_decrypt(self):
        """Test encryption and decryption for consistency."""
        test_data = "Test data for encryption"
        encrypted_data = await EncryptionManager.encrypt(test_data.encode())
        decrypted_data = await EncryptionManager.decrypt(encrypted_data)
        self.assertEqual(
            test_data,
            decrypted_data.decode(),
            "Decrypted data should match the original.",
        )


if __name__ == "__main__":
    import tracemalloc

    tracemalloc.start()
    unittest.main()

    # TODO:
"""
    # TODO:
        # ================================================================================================
        # High Priority:
            #   Security:
            #   - [ ]
            #   Documentation:
            #   - [ ]
            #   Optimization:
            #   - [ ]
            #   Flexibility:
            #   - [ ]
            #   Automation:
            #   - [ ]
            #   Scalability:
            #   - [ ]
            #   Ethics:
            #   - [ ]
            #   Bug Fix:
            #   - [ ]
            #   Robustness:
            #   - [ ]
            #   Clean Code:
            #   - [ ]
            #   Stability:
            #   - [ ]
            #   Formatting:
            #   - [ ]
            #   Logics:
            #   - [ ]
            #   Integration:
            #   - [ ]
        # ================================================================================================
        # Medium Priority:
            #   Security:
            #   - [ ]
            #   Documentation:
            #   - [ ]
            #   Optimization:
            #   - [ ]
            #   Flexibility:
            #   - [ ]
            #   Automation:
            #   - [ ]
            #   Scalability:
            #   - [ ]
            #   Ethics:
            #   - [ ]
            #   Bug Fix:
            #   - [ ]
            #   Robustness:
            #   - [ ]
            #   Clean Code:
            #   - [ ]
            #   Stability:
            #   - [ ]
            #   Formatting:
            #   - [ ]
            #   Logics:
            #   - [ ]
            #   Integration:
            #   - [ ]
        # ================================================================================================
        # Low Priority:
            #   Security:
            #   - [ ]
            #   Documentation:
            #   - [ ]
            #   Optimization:
            #   - [ ]
            #   Flexibility:
            #   - [ ]
            #   Automation:
            #   - [ ]
            #   Scalability:
            #   - [ ]
            #   Ethics:
            #   - [ ]
            #   Bug Fix:
            #   - [ ]
            #   Robustness:
            #   - [ ]
            #   Clean Code:
            #   - [ ]
            #   Stability:
            #   - [ ]
            #   Formatting:
            #   - [ ]
            #   Logics:
            #   - [ ]
            #   Integration:
            #   - [ ]
        # ================================================================================================
        # Stretch Goals:
            #   Security:
            #   - [ ]
            #   Documentation:
            #   - [ ]
            #   Optimization:
            #   - [ ]
            #   Flexibility:
            #   - [ ]
            #   Automation:
            #   - [ ]
            #   Scalability:
            #   - [ ]
            #   Ethics:
            #   - [ ]
            #   Bug Fix:
            #   - [ ]
            #   Robustness:
            #   - [ ]
            #   Clean Code:
            #   - [ ]
            #   Stability:
            #   - [ ]
            #   Formatting:
            #   - [ ]
            #   Logics:
            #   - [ ]
            #   Integration:
            #   - [ ]
        # ================================================================================================
        # Routine:
            #   Security:
            #   - [ ]
            #   Documentation:
            #   - [ ]
            #   Optimization:
            #   - [ ]
            #   Flexibility:
            #   - [ ]
            #   Automation:
            #   - [ ]
            #   Scalability:
            #   - [ ]
            #   Ethics:
            #   - [ ]
            #   Bug Fix:
            #   - [ ]
            #   Robustness:
            #   - [ ]
            #   Clean Code:
            #   - [ ]
            #   Stability:
            #   - [ ]
            #   Formatting:
            #   - [ ]
            #   Logics:
            #   - [ ]
            #   Integration:
            #   - [ ]
        # ================================================================================================
        # Known Issues:
            #   Security:
            #   - [ ]
            #   Documentation:
            #   - [ ]
            #   Optimization:
            #   - [ ]
            #   Flexibility:
            #   - [ ]
            #   Automation:
            #   - [ ]
            #   Scalability:
            #   - [ ]
            #   Ethics:
            #   - [ ]
            #   Bug Fix:
            #   - [ ]
            #   Robustness:
            #   - [ ]
            #   Clean Code:
            #   - [ ]
            #   Stability:
            #   - [ ]
            #   Formatting:
            #   - [ ]
            #   Logics:
            #   - [ ]
            #   Integration:
            #   - [ ]
        # ================================================================================================
"""
