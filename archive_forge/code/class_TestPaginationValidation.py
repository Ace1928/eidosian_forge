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
class TestPaginationValidation(unittest.TestCase):

    def test_validate_pagination_params_success(self):
        validate_pagination_params(0, 10)

    def test_validate_pagination_params_failure(self):
        with self.assertRaises(ValueError):
            validate_pagination_params(-1, 10)