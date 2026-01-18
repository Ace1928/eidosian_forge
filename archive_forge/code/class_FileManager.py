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
class FileManager:
    """Manager for acquiring and closing a file object.

    Use FileManager subclasses (CachingFileManager in particular) on backend
    storage classes to automatically handle issues related to keeping track of
    many open files and transferring them between multiple processes.
    """

    def acquire(self, needs_lock=True):
        """Acquire the file object from this manager."""
        raise NotImplementedError()

    def acquire_context(self, needs_lock=True):
        """Context manager for acquiring a file. Yields a file object.

        The context manager unwinds any actions taken as part of acquisition
        (i.e., removes it from any cache) if an exception is raised from the
        context. It *does not* automatically close the file.
        """
        raise NotImplementedError()

    def close(self, needs_lock=True):
        """Close the file object associated with this manager, if needed."""
        raise NotImplementedError()