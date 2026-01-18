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
def _dir_travel(path: Path, excludes: List[Callable], handler: Callable, logger: Optional[logging.Logger]=default_logger):
    """Travels the path recursively, calling the handler on each subpath.

    Respects excludes, which will be called to check if this path is skipped.
    """
    e = _get_gitignore(path)
    if e is not None:
        excludes.append(e)
    skip = any((e(path) for e in excludes))
    if not skip:
        try:
            handler(path)
        except Exception as e:
            logger.error(f'Issue with path: {path}')
            raise e
        if path.is_dir():
            for sub_path in path.iterdir():
                _dir_travel(sub_path, excludes, handler, logger=logger)
    if e is not None:
        excludes.pop()