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
def _create_validation_file(self):
    """On the creation of a storage context, create a validation file at the
        storage path to verify that the storage path can be written to.
        This validation file is also used to check whether the storage path is
        accessible by all nodes in the cluster."""
    valid_file = os.path.join(self.experiment_fs_path, _VALIDATE_STORAGE_MARKER_FILENAME)
    self.storage_filesystem.create_dir(self.experiment_fs_path)
    with self.storage_filesystem.open_output_stream(valid_file):
        pass