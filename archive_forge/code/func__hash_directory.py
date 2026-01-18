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
def _hash_directory(root: Path, relative_path: Path, excludes: Optional[Callable], logger: Optional[logging.Logger]=default_logger) -> bytes:
    """Helper function to create hash of a directory.

    It'll go through all the files in the directory and xor
    hash(file_name, file_content) to create a hash value.
    """
    hash_val = b'0' * 8
    BUF_SIZE = 4096 * 1024

    def handler(path: Path):
        md5 = hashlib.md5()
        md5.update(str(path.relative_to(relative_path)).encode())
        if not path.is_dir():
            try:
                f = path.open('rb')
            except Exception as e:
                logger.debug(f'Skipping contents of file {path} when calculating package hash because the file could not be opened: {e}')
            else:
                try:
                    data = f.read(BUF_SIZE)
                    while len(data) != 0:
                        md5.update(data)
                        data = f.read(BUF_SIZE)
                finally:
                    f.close()
        nonlocal hash_val
        hash_val = _xor_bytes(hash_val, md5.digest())
    excludes = [] if excludes is None else [excludes]
    _dir_travel(root, excludes, handler, logger=logger)
    return hash_val