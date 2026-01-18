"""
Logging Configuration Module

This module configures logging for the Image Interconversion GUI application, directing logs to both a file and the console to facilitate effective monitoring and debugging. It supports dynamic log level changes through integration with the config_manager module, enhancing flexibility for different environments.

Author: Lloyd Handyside
Creation Date: 2024-04-08
Last Modified: 2024-04-08
Version: 1.0.0
"""

import logging
from logging.handlers import RotatingFileHandler
from config_manager import (
    ConfigManager,
)  # Integration with config_manager for dynamic log level configuration

__all__ = [
    "configure_logging",
    "log_function_call",
]  # Updated to include the log_function_call decorator


def log_function_call(func):
    """
    A decorator that logs the entry and exit points of the decorated function.

    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: The wrapped function with logging.
    """

    def wrapper(*args, **kwargs):
        logging.debug(f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        logging.debug(f"Exiting {func.__name__}")
        return result

    return wrapper


@log_function_call  # Decorator applied to enhance logging
def configure_logging() -> None:
    """
    Configures the global logging level, format, and handlers for the application based on settings from the config.ini file.

    Sets up logging to output to both a file named 'image_interconversion_gui.log' and the console. It aims to be invoked at the application's startup phase.

    No parameters.
    Returns: None.

    Example:
        configure_logging()
    """
    try:
        handlers = [
            RotatingFileHandler(
                "image_interconversion_gui.log",
                maxBytes=10485760,
                backupCount=5,
                mode="a",
            ),
            logging.StreamHandler(),
        ]
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )
        logging.info("Logging configuration successfully set up.")
    except Exception as e:
        logging.error(f"Failed to configure logging: {e}")


# TODO:
# - Introduce configurable logging levels for different application components, utilising the config_manager.py file to centralise configuration to the main config.ini file.

# Known Issues:
# - None
