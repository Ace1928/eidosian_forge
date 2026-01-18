from __future__ import annotations
import atexit
import contextlib
import io
import threading
import uuid
import warnings
from collections.abc import Hashable
from typing import Any
from xarray.backends.locks import acquire
from xarray.backends.lru_cache import LRUCache
from xarray.core import utils
from xarray.core.options import OPTIONS
@atexit.register
def _remove_del_method():
    del CachingFileManager.__del__