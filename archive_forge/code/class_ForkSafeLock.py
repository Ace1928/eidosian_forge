from __future__ import nested_scopes
import platform
import weakref
import struct
import warnings
import functools
from contextlib import contextmanager
import sys  # Note: the sys import must be here anyways (others depend on it)
import codecs as _codecs
import os
from _pydevd_bundle import pydevd_vm_type
from _pydev_bundle._pydev_saved_modules import thread, threading
class ForkSafeLock(object):
    """
        A lock which is fork-safe (when a fork is done, `pydevd_constants.after_fork()`
        should be called to reset the locks in the new process to avoid deadlocks
        from a lock which was locked during the fork).

        Note:
            Unlike `threading.Lock` this class is not completely atomic, so, doing:

            lock = ForkSafeLock()
            with lock:
                ...

            is different than using `threading.Lock` directly because the tracing may
            find an additional function call on `__enter__` and on `__exit__`, so, it's
            not recommended to use this in all places, only where the forking may be important
            (so, for instance, the locks on PyDB should not be changed to this lock because
            of that -- and those should all be collected in the new process because PyDB itself
            should be completely cleared anyways).

            It's possible to overcome this limitation by using `ForkSafeLock.acquire` and
            `ForkSafeLock.release` instead of the context manager (as acquire/release are
            bound to the original implementation, whereas __enter__/__exit__ is not due to Python
            limitations).
        """

    def __init__(self, rlock=False):
        self._rlock = rlock
        self._init()
        _fork_safe_locks.append(weakref.ref(self))

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._lock.__exit__(exc_type, exc_val, exc_tb)

    def _init(self):
        if self._rlock:
            self._lock = threading.RLock()
        else:
            self._lock = thread.allocate_lock()
        self.acquire = self._lock.acquire
        self.release = self._lock.release
        _fork_safe_locks.append(weakref.ref(self))