import dataclasses
import fnmatch
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Type, Union
from ray._private.storage import _get_storage_uri
from ray.air._internal.filelock import TempFileLock
from ray.train._internal.syncer import SyncConfig, Syncer, _BackgroundSyncer
from ray.train.constants import _get_defaults_results_dir
def _is_directory(fs: pyarrow.fs.FileSystem, fs_path: str) -> bool:
    """Checks if (fs, fs_path) is a directory or a file.

    Raises:
        FileNotFoundError: if (fs, fs_path) doesn't exist.
    """
    file_info = fs.get_file_info(fs_path)
    if file_info.type == pyarrow.fs.FileType.NotFound:
        raise FileNotFoundError(f'Path not found: ({fs}, {fs_path})')
    return not file_info.is_file