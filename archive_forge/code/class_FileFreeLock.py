import os
import types
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pyarrow as pa
from filelock import BaseFileLock, Timeout
from . import config
from .arrow_dataset import Dataset
from .arrow_reader import ArrowReader
from .arrow_writer import ArrowWriter
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadManager
from .features import Features
from .info import DatasetInfo, MetricInfo
from .naming import camelcase_to_snakecase
from .utils._filelock import FileLock
from .utils.deprecation_utils import deprecated
from .utils.logging import get_logger
from .utils.py_utils import copyfunc, temp_seed
class FileFreeLock(BaseFileLock):
    """Thread lock until a file **cannot** be locked"""

    def __init__(self, lock_file, *args, **kwargs):
        self.filelock = FileLock(lock_file)
        super().__init__(self.filelock.lock_file, *args, **kwargs)

    def _acquire(self):
        try:
            self.filelock.acquire(timeout=0.01, poll_intervall=0.02)
        except Timeout:
            self._context.lock_file_fd = self.filelock.lock_file
        else:
            self.filelock.release()
            self._context.lock_file_fd = None

    def _release(self):
        self._context.lock_file_fd = None