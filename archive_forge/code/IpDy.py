"""
Config Manager Module

This module provides a flexible and robust configuration manager designed to handle application settings across various modules and programs. It supports loading, retrieving, setting, and saving configuration values with ease.

Author: Your Name
Creation Date: YYYY-MM-DD
Last Modified: YYYY-MM-DD

Functionalities:
- Load and manage multiple configuration files.
- Retrieve and set configuration values with type conversion and fallback support.
- Save configuration changes back to files.

"""

import configparser
from typing import Any, Dict, Optional, Union
import os
import logging


class ConfigManager:
    """
    A flexible and robust configuration manager to handle application settings across various modules and programs.

    Attributes:
        config_files (Dict[str, configparser.ConfigParser]): A dictionary mapping configuration file names to their parser instances.
    """

    def __init__(self):
        """
        Initializes the ConfigManager with an empty dictionary for config files.
        """
        self.config_files: Dict[str, configparser.ConfigParser] = {}

    def load_config(self, file_path: str, config_name: Optional[str] = None) -> None:
        """
        Loads a configuration file into the manager.

        Args:
            file_path (str): The path to the configuration file.
            config_name (Optional[str]): An optional name to reference the configuration file. Defaults to the file name if not provided.
        """
        try:
            if not config_name:
                config_name = os.path.basename(file_path)

            config_parser = configparser.ConfigParser()
            config_parser.read(file_path)
            self.config_files[config_name] = config_parser
        except Exception as e:
            logging.error(f"Failed to load configuration from {file_path}: {e}")

    def get(
        self,
        config_name: str,
        section: str,
        option: str,
        fallback: Optional[Any] = None,
    ) -> Union[str, int, float, bool, None]:
        """
        Retrieves a configuration value with support for fallbacks.

        Args:
            config_name (str): The name of the configuration file.
            section (str): The section in the configuration file.
            option (str): The option key to retrieve the value for.
            fallback (Optional[Any]): The fallback value if the option is not found.

        Returns:
            Union[str, int, float, bool, None]: The value of the configuration option, converted to the appropriate type.

        Example:
            >>> config_manager.get("main_config", "Database", "host", fallback="localhost")
            'db.example.com'
        """
        config_parser = self.config_files.get(config_name)
        if config_parser and config_parser.has_option(section, option):
            value = config_parser.get(section, option)
            # Convert value to int or float if possible, fallback to bool or raw string
            for cast in (int, float):
                try:
                    return cast(value)
                except ValueError:
                    continue
            if value.lower() in ("true", "false"):
                return value.lower() == "true"
            return value
        return fallback

    def set(self, config_name: str, section: str, option: str, value: Any) -> None:
        """
        Sets a configuration value.

        Args:
            config_name (str): The name of the configuration file.
            section (str): The section in the configuration file.
            option (str): The option key to set the value for.
            value (Any): The value to set.
        """
        config_parser = self.config_files.get(config_name)
        if not config_parser:
            raise ValueError(f"Configuration '{config_name}' not loaded.")
        if not config_parser.has_section(section):
            config_parser.add_section(section)
        config_parser.set(section, option, str(value))

    def save_config(self, config_name: str, file_path: Optional[str] = None) -> None:
        """
        Saves the specified configuration back to a file.

        Args:
            config_name (str): The name of the configuration to save.
            file_path (Optional[str]): The file path to save the configuration to. Uses the original path if not provided.
        """
        if not file_path:
            file_path = config_name  # Assuming config_name was the file path if no explicit file_path is provided
        config_parser = self.config_files.get(config_name)
        if not config_parser:
            raise ValueError(f"Configuration '{config_name}' not loaded.")
        with open(file_path, "w") as configfile:
            config_parser.write(configfile)


__all__ = ["ConfigManager"]

# Example usage:
config_manager = ConfigManager()
config_manager.load_config("config.ini", "main_config")
print(config_manager.get("main_config", "Database", "host", fallback="localhost"))
config_manager.set("main_config", "Database", "port", 5432)
config_manager.save_config("main_config")

# TODO:
# - Implement encryption for sensitive configuration values.
# - Add support for JSON-based configuration files.
# - Improve error handling for file access and parsing issues.

# Known Issues:
# - None identified at this time.
