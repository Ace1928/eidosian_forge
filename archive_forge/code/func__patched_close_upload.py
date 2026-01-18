import abc
from collections import defaultdict
import collections.abc
from contextlib import contextmanager
import os
from pathlib import (  # type: ignore
import shutil
import sys
from typing import (
from urllib.parse import urlparse
from warnings import warn
from cloudpathlib.enums import FileCacheMode
from . import anypath
from .exceptions import (
def _patched_close_upload(*args, **kwargs) -> None:
    wrapped_close(*args, **kwargs)
    if not self._dirty:
        return
    if self._local.stat().st_mtime < original_mtime:
        new_mtime = original_mtime + 1
        os.utime(self._local, times=(new_mtime, new_mtime))
    self._upload_local_to_cloud(force_overwrite_to_cloud=force_overwrite_to_cloud)
    self._dirty = False