"""
Module: config.py

This module defines the configuration settings for the trading bot application. It utilizes environment variables to ensure sensitive information is not hard-coded into the application, aligning with best practices for security and configuration management.

Dependencies:
- os: To access environment variables.

Classes:
- Config: Contains configuration settings for the application.

Authorship and Versioning Details:
    Author: Lloyd Handyside
    Creation Date: 2024-04-16 (ISO 8601 Format)
    Last Modified: 2024-04-16 (ISO 8601 Format)
    Version: 1.0.0 (Semantic Versioning)
    Contact: lloyd.handyside@neuroforge.io
    Ownership: Neuro Forge
    Status: Draft (Subject to change)
"""

import os
from typing import Optional


class Config:
    """
    Configuration class containing all the settings for the application, sourced from environment variables.

    Attributes:
        JWT_SECRET_KEY (str): Secret key for JWT token generation and verification. Falls back to a default if not set.
        DATABASE_URI (Optional[str]): URI for the database connection. None if not set, indicating in-memory or default storage should be used.
        DEBUG_MODE (bool): Flag to indicate if the application should run in debug mode. Defaults to False.
    """

    JWT_SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY", "your_jwt_secret_key")
    DATABASE_URI: Optional[str] = os.environ.get("DATABASE_URI")
    DEBUG_MODE: bool = os.environ.get("DEBUG_MODE", "False").lower() in [
        "true",
        "1",
        "t",
    ]

    # Additional configuration parameters can be added here, following the same pattern.
