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
def _get_pointer_path(storage_folder: str, revision: str, relative_filename: str) -> str:
    snapshot_path = os.path.join(storage_folder, 'snapshots')
    pointer_path = os.path.join(snapshot_path, revision, relative_filename)
    if Path(os.path.abspath(snapshot_path)) not in Path(os.path.abspath(pointer_path)).parents:
        raise ValueError(f"Invalid pointer path: cannot create pointer path in snapshot folder if `storage_folder='{storage_folder}'`, `revision='{revision}'` and `relative_filename='{relative_filename}'`.")
    return pointer_path