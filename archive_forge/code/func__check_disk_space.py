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
def _check_disk_space(expected_size: int, target_dir: Union[str, Path]) -> None:
    """Check disk usage and log a warning if there is not enough disk space to download the file.

    Args:
        expected_size (`int`):
            The expected size of the file in bytes.
        target_dir (`str`):
            The directory where the file will be stored after downloading.
    """
    target_dir = Path(target_dir)
    for path in [target_dir] + list(target_dir.parents):
        try:
            target_dir_free = shutil.disk_usage(path).free
            if target_dir_free < expected_size:
                warnings.warn(f'Not enough free disk space to download the file. The expected file size is: {expected_size / 1000000.0:.2f} MB. The target location {target_dir} only has {target_dir_free / 1000000.0:.2f} MB free disk space.')
            return
        except OSError:
            pass