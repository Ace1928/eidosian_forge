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
class DummyFileManager(FileManager):
    """FileManager that simply wraps an open file in the FileManager interface."""

    def __init__(self, value):
        self._value = value

    def acquire(self, needs_lock=True):
        del needs_lock
        return self._value

    @contextlib.contextmanager
    def acquire_context(self, needs_lock=True):
        del needs_lock
        yield self._value

    def close(self, needs_lock=True):
        del needs_lock
        self._value.close()