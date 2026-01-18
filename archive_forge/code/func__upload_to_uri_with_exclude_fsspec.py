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
def _upload_to_uri_with_exclude_fsspec(local_path: str, fs: 'pyarrow.fs', fs_path: str, exclude: Optional[List[str]]) -> None:
    local_fs = _ExcludingLocalFilesystem(root_path=local_path, exclude=exclude)
    handler = pyarrow.fs.FSSpecHandler(local_fs)
    source_fs = pyarrow.fs.PyFileSystem(handler)
    _create_directory(fs=fs, fs_path=fs_path)
    _pyarrow_fs_copy_files(local_path, fs_path, source_filesystem=source_fs, destination_filesystem=fs)