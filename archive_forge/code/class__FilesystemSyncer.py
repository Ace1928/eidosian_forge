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
class _FilesystemSyncer(_BackgroundSyncer):
    """Syncer between local filesystem and a `storage_filesystem`."""

    def __init__(self, storage_filesystem: Optional['pyarrow.fs.FileSystem'], **kwargs):
        self.storage_filesystem = storage_filesystem
        super().__init__(**kwargs)

    def _sync_up_command(self, local_path: str, uri: str, exclude: Optional[List]=None) -> Tuple[Callable, Dict]:
        fs_path = uri
        return (_upload_to_fs_path, dict(local_path=local_path, fs=self.storage_filesystem, fs_path=fs_path, exclude=exclude))

    def _sync_down_command(self, uri: str, local_path: str) -> Tuple[Callable, Dict]:
        fs_path = uri
        return (_download_from_fs_path, dict(fs=self.storage_filesystem, fs_path=fs_path, local_path=local_path))

    def _delete_command(self, uri: str) -> Tuple[Callable, Dict]:
        fs_path = uri
        return (_delete_fs_path, dict(fs=self.storage_filesystem, fs_path=fs_path))