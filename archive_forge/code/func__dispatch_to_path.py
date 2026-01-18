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
def _dispatch_to_path(self, func: str, *args, **kwargs) -> Any:
    """Some functions we can just dispatch to the pathlib version
        We want to do this explicitly so we don't have to support all
        of pathlib and subclasses can override individually if necessary.
        """
    path_version = self._path.__getattribute__(func)
    if callable(path_version):
        path_version = path_version(*args, **kwargs)
    if isinstance(path_version, PurePosixPath):
        path_version = _resolve(path_version)
        return self._new_cloudpath(path_version)
    if isinstance(path_version, collections.abc.Sequence) and len(path_version) > 0 and isinstance(path_version[0], PurePosixPath):
        sequence_class = type(path_version) if not isinstance(path_version, _PathParents) else tuple
        return sequence_class((self._new_cloudpath(_resolve(p)) for p in path_version if _resolve(p) != p.root))
    else:
        return path_version