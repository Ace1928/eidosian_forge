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
def _upload_to_fs_path(local_path: str, fs: pyarrow.fs.FileSystem, fs_path: str, exclude: Optional[List[str]]=None) -> None:
    """Uploads a local directory or file to (fs, fs_path).

    NOTE: This will create all necessary parent directories at the destination.

    Args:
        local_path: The local path to upload.
        fs: The filesystem to upload to.
        fs_path: The filesystem path where the dir/file will be uploaded to.
        exclude: A list of filename matches to exclude from upload. This includes
            all files under subdirectories as well.
            This pattern will match with the relative paths of all files under
            `local_path`.
            Ex: ["*.png"] to exclude all .png images.
    """
    if not exclude:
        _create_directory(fs=fs, fs_path=fs_path)
        _pyarrow_fs_copy_files(local_path, fs_path, destination_filesystem=fs)
        return
    _upload_to_uri_with_exclude_fsspec(local_path=local_path, fs=fs, fs_path=fs_path, exclude=exclude)