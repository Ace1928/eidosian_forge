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
def _delete_fs_path(fs: pyarrow.fs.FileSystem, fs_path: str):
    is_dir = _is_directory(fs, fs_path)
    try:
        if is_dir:
            fs.delete_dir(fs_path)
        else:
            fs.delete_file(fs_path)
    except Exception:
        logger.exception(f'Caught exception when deleting path at ({fs}, {fs_path}):')