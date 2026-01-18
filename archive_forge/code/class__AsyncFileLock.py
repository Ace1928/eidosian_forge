import asyncio
import hashlib
import logging
import os
import shutil
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlparse
from zipfile import ZipFile
from filelock import FileLock
from ray.util.annotations import DeveloperAPI
from ray._private.ray_constants import (
from ray._private.runtime_env.conda_utils import exec_cmd_stream_to_logger
from ray._private.thirdparty.pathspec import PathSpec
from ray.experimental.internal_kv import (
class _AsyncFileLock:
    """Asyncio version used to prevent blocking event loop."""

    def __init__(self, lock_file: str):
        self.file = FileLock(lock_file)

    async def __aenter__(self):
        while True:
            try:
                self.file.acquire(timeout=0)
                return
            except TimeoutError:
                await asyncio.sleep(0.1)

    async def __aexit__(self, exc_type, exc, tb):
        self.file.release()