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
class _RefCounter:
    """Class for keeping track of reference counts."""

    def __init__(self, counts):
        self._counts = counts
        self._lock = threading.Lock()

    def increment(self, name):
        with self._lock:
            count = self._counts[name] = self._counts.get(name, 0) + 1
        return count

    def decrement(self, name):
        with self._lock:
            count = self._counts[name] - 1
            if count:
                self._counts[name] = count
            else:
                del self._counts[name]
        return count