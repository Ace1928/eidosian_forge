from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import rmtree
from typing import List
from tornado.concurrent import run_on_executor
from tornado.gen import convert_yielded
from .manager import lsp_message_listener
from .paths import file_uri_to_path, is_relative
from .types import LanguageServerManagerAPI
class ShadowFilesystemError(ValueError):
    """Error in the shadow file system."""