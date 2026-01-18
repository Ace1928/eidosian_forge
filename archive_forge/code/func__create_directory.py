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
def _create_directory(fs: pyarrow.fs.FileSystem, fs_path: str) -> None:
    """Create directory at (fs, fs_path).

    Some external filesystems require directories to already exist, or at least
    the `netloc` to be created (e.g. PyArrows ``mock://`` filesystem).

    Generally this should be done before and outside of Ray applications. This
    utility is thus primarily used in testing, e.g. of ``mock://` URIs.
    """
    try:
        fs.create_dir(fs_path)
    except Exception:
        logger.exception(f'Caught exception when creating directory at ({fs}, {fs_path}):')