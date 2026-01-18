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
def _chmod_and_replace(src: str, dst: str) -> None:
    """Set correct permission before moving a blob from tmp directory to cache dir.

    Do not take into account the `umask` from the process as there is no convenient way
    to get it that is thread-safe.

    See:
    - About umask: https://docs.python.org/3/library/os.html#os.umask
    - Thread-safety: https://stackoverflow.com/a/70343066
    - About solution: https://github.com/huggingface/huggingface_hub/pull/1220#issuecomment-1326211591
    - Fix issue: https://github.com/huggingface/huggingface_hub/issues/1141
    - Fix issue: https://github.com/huggingface/huggingface_hub/issues/1215
    """
    tmp_file = Path(dst).parent.parent / f'tmp_{uuid.uuid4()}'
    try:
        tmp_file.touch()
        cache_dir_mode = Path(tmp_file).stat().st_mode
        os.chmod(src, stat.S_IMODE(cache_dir_mode))
    finally:
        tmp_file.unlink()
    shutil.move(src, dst)