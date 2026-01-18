import copy
import fnmatch
import inspect
import io
import json
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO, Dict, Generator, Literal, Optional, Tuple, Union
from urllib.parse import quote, urlparse
import requests
from filelock import FileLock
from huggingface_hub import constants
from . import __version__  # noqa: F401 # for backward compatibility
from .constants import (
from .utils import (
from .utils._deprecation import _deprecate_method
from .utils._headers import _http_user_agent
from .utils._runtime import _PY_VERSION  # noqa: F401 # for backward compatibility
from .utils._typing import HTTP_METHOD_T
from .utils.insecure_hashlib import sha256
def _to_local_dir(path: str, local_dir: str, relative_filename: str, use_symlinks: Union[bool, Literal['auto']]) -> str:
    """Place a file in a local dir (different than cache_dir).

    Either symlink to blob file in cache or duplicate file depending on `use_symlinks` and file size.
    """
    local_dir_filepath = os.path.join(local_dir, relative_filename)
    if Path(os.path.abspath(local_dir)) not in Path(os.path.abspath(local_dir_filepath)).parents:
        raise ValueError(f"Cannot copy file '{relative_filename}' to local dir '{local_dir}': file would not be in the local directory.")
    os.makedirs(os.path.dirname(local_dir_filepath), exist_ok=True)
    real_blob_path = os.path.realpath(path)
    if use_symlinks == 'auto':
        use_symlinks = os.stat(real_blob_path).st_size > constants.HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD
    if use_symlinks:
        _create_symlink(real_blob_path, local_dir_filepath, new_blob=False)
    else:
        shutil.copyfile(real_blob_path, local_dir_filepath)
    return local_dir_filepath