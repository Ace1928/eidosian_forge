import io
import os
import socket
import threading
import time
import selectors
from contextlib import suppress
from . import errors
from ._compat import IS_WINDOWS
from .makefile import MakeFile
class _ThreadsafeSelector:
    """Thread-safe wrapper around a DefaultSelector.

    There are 2 thread contexts in which it may be accessed:
      * the selector thread
      * one of the worker threads in workers/threadpool.py

    The expected read/write patterns are:
      * :py:func:`~iter`: selector thread
      * :py:meth:`register`: selector thread and threadpool,
        via :py:meth:`~cheroot.workers.threadpool.ThreadPool.put`
      * :py:meth:`unregister`: selector thread only

    Notably, this means :py:class:`_ThreadsafeSelector` never needs to worry
    that connections will be removed behind its back.

    The lock is held when iterating or modifying the selector but is not
    required when :py:meth:`select()ing <selectors.BaseSelector.select>` on it.
    """

    def __init__(self):
        self._selector = selectors.DefaultSelector()
        self._lock = threading.Lock()

    def __len__(self):
        with self._lock:
            return len(self._selector.get_map() or {})

    @property
    def connections(self):
        """Retrieve connections registered with the selector."""
        with self._lock:
            mapping = self._selector.get_map() or {}
            for _, (_, sock_fd, _, conn) in mapping.items():
                yield (sock_fd, conn)

    def register(self, fileobj, events, data=None):
        """Register ``fileobj`` with the selector."""
        with self._lock:
            return self._selector.register(fileobj, events, data)

    def unregister(self, fileobj):
        """Unregister ``fileobj`` from the selector."""
        with self._lock:
            return self._selector.unregister(fileobj)

    def select(self, timeout=None):
        """Return socket fd and data pairs from selectors.select call.

        Returns entries ready to read in the form:
            (socket_file_descriptor, connection)
        """
        return ((key.fd, key.data) for key, _ in self._selector.select(timeout=timeout))

    def close(self):
        """Close the selector."""
        with self._lock:
            self._selector.close()