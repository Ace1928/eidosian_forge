"""
Logging Configuration for Image Interconversion GUI

This module configures logging for the Image Interconversion GUI application, directing logs to both a file and the console to facilitate effective monitoring and debugging.

Author: Lloyd Handyside
Creation Date: 2024-04-08
Last Modified: 2024-04-08
"""

import logging
from logging.handlers import RotatingFileHandler

__all__ = ["configure_logging"]


def configure_logging() -> None:
    """
    Configures the global logging level, format, and handlers for the application.

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
