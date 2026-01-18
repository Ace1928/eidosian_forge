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
def _get_del_lock_path(path: str, suffix: str=None) -> str:
    """Get the path to the deletion lock file for a file/directory at `path`.

    Example:

        >>> _get_del_lock_path("/tmp/checkpoint_tmp")  # doctest: +ELLIPSIS
        '/tmp/checkpoint_tmp.del_lock_...
        >>> _get_del_lock_path("/tmp/checkpoint_tmp/")  # doctest: +ELLIPSIS
        '/tmp/checkpoint_tmp.del_lock_...
        >>> _get_del_lock_path("/tmp/checkpoint_tmp.txt")  # doctest: +ELLIPSIS
        '/tmp/checkpoint_tmp.txt.del_lock_...

    """
    suffix = suffix if suffix is not None else str(os.getpid())
    return f'{path.rstrip('/')}.del_lock_{suffix}'