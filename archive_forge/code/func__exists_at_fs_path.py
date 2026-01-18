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
def _exists_at_fs_path(fs: pyarrow.fs.FileSystem, fs_path: str) -> bool:
    """Returns True if (fs, fs_path) exists."""
    valid = fs.get_file_info(fs_path)
    return valid.type != pyarrow.fs.FileType.NotFound