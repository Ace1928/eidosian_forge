"""
Logging Configuration for Image Interconversion GUI

This module provides the logging configuration used throughout the Image Interconversion GUI application. It sets up logging to both a file and the console, ensuring that application behavior can be monitored and debugged effectively.

Author: [Author Name]
Creation Date: [Creation Date]
"""

import logging

__all__ = ["configure_logging"]


def configure_logging() -> None:
    """
    Configures the global logging level, format, and handlers for the application.

    This function sets up logging to output to both a file named 'image_interconversion_gui.log' and the console. It is designed to be called at the application startup.

    No parameters.
    Returns: None.
    """
    try:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("image_interconversion_gui.log", mode="a"),
                logging.StreamHandler(),  # Console logging
            ],
        )
        logging.info("Logging configuration successfully set up.")
    except Exception as e:
        logging.error(f"Failed to configure logging: {e}")


# TODO:
# - Implement rotation of log files to avoid excessive file size.
# - Explore more granular logging levels for different parts of the application.

# Known Issues:
# - None identified at this time.
