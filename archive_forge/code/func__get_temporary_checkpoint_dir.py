import contextlib
import glob
import json
import logging
import os
import platform
import shutil
import tempfile
import traceback
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union
import pyarrow.fs
from ray.air._internal.filelock import TempFileLock
from ray.train._internal.storage import _download_from_fs_path, _exists_at_fs_path
from ray.util.annotations import PublicAPI
def _get_temporary_checkpoint_dir(self) -> str:
    """Return the name for the temporary checkpoint dir that this checkpoint
        will get downloaded to, if accessing via `to_directory` or `as_directory`.
        """
    tmp_dir_path = tempfile.gettempdir()
    checkpoint_dir_name = _CHECKPOINT_TEMP_DIR_PREFIX + self._uuid.hex
    if platform.system() == 'Windows':
        del_lock_name = _get_del_lock_path('')
        checkpoint_dir_name = _CHECKPOINT_TEMP_DIR_PREFIX + self._uuid.hex[-259 + len(_CHECKPOINT_TEMP_DIR_PREFIX) + len(tmp_dir_path) + len(del_lock_name):]
        if not checkpoint_dir_name.startswith(_CHECKPOINT_TEMP_DIR_PREFIX):
            raise RuntimeError("Couldn't create checkpoint directory due to length constraints. Try specifying a shorter checkpoint path.")
    return os.path.join(tmp_dir_path, checkpoint_dir_name)