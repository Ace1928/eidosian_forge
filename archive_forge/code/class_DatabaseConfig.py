import asyncio
import aiosqlite
from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv
from typing import Optional, List, Tuple, AsyncContextManager, NoReturn, AsyncGenerator
import backoff
from contextlib import asynccontextmanager
import io
from PIL import Image  # Import PIL.Image for image handling
import unittest
from unittest import IsolatedAsyncioTestCase
from core_services import ConfigManager, LoggingManager, EncryptionManager
from image_processing import (
class DatabaseConfig:
    """
    Holds the configuration for the database connection and encryption key file path.

    Attributes:
        DB_PATH (str): Path to the SQLite database file.
        KEY_FILE (str): Path to the encryption key file.
        CONFIG_PATH: str = "config.ini"
    """
    DB_PATH: str = 'image_db.sqlite'
    KEY_FILE: str = 'encryption.key'
    CONFIG_PATH: str = 'config.ini'