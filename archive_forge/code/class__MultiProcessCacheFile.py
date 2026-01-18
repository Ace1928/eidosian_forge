from __future__ import print_function
import argparse
import contextlib
import datetime
import json
import os
import threading
import warnings
import httplib2
import oauth2client
import oauth2client.client
from oauth2client import service_account
from oauth2client import tools  # for gflags declarations
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.py import exceptions
from apitools.base.py import util
class _MultiProcessCacheFile(object):
    """Simple multithreading and multiprocessing safe cache file.

    Notes on behavior:
    * the fasteners.InterProcessLock object cannot reliably prevent threads
      from double-acquiring a lock. A threading lock is used in addition to
      the InterProcessLock. The threading lock is always acquired first and
      released last.
    * The interprocess lock will not deadlock. If a process can not acquire
      the interprocess lock within `_lock_timeout` the call will return as
      a cache miss or unsuccessful cache write.
    * App Engine environments cannot be process locked because (1) the runtime
      does not provide monotonic time and (2) different processes may or may
      not share the same machine. Because of this, process locks are disabled
      and locking is only guaranteed to protect against multithreaded access.
    """
    _lock_timeout = 1
    _encoding = 'utf-8'
    _thread_lock = threading.Lock()

    def __init__(self, filename):
        self._file = None
        self._filename = filename
        if _FASTENERS_AVAILABLE:
            self._process_lock_getter = self._ProcessLockAcquired
            self._process_lock = fasteners.InterProcessLock('{0}.lock'.format(filename))
        else:
            self._process_lock_getter = self._DummyLockAcquired
            self._process_lock = None

    @contextlib.contextmanager
    def _ProcessLockAcquired(self):
        """Context manager for process locks with timeout."""
        try:
            is_locked = self._process_lock.acquire(timeout=self._lock_timeout)
            yield is_locked
        finally:
            if is_locked:
                self._process_lock.release()

    @contextlib.contextmanager
    def _DummyLockAcquired(self):
        """Lock context manager for environments without process locks."""
        yield True

    def LockedRead(self):
        """Acquire an interprocess lock and dump cache contents.

        This method safely acquires the locks then reads a string
        from the cache file. If the file does not exist and cannot
        be created, it will return None. If the locks cannot be
        acquired, this will also return None.

        Returns:
          cache data - string if present, None on failure.
        """
        file_contents = None
        with self._thread_lock:
            if not self._EnsureFileExists():
                return None
            with self._process_lock_getter() as acquired_plock:
                if not acquired_plock:
                    return None
                with open(self._filename, 'rb') as f:
                    file_contents = f.read().decode(encoding=self._encoding)
        return file_contents

    def LockedWrite(self, cache_data):
        """Acquire an interprocess lock and write a string.

        This method safely acquires the locks then writes a string
        to the cache file. If the string is written successfully
        the function will return True, if the write fails for any
        reason it will return False.

        Args:
          cache_data: string or bytes to write.

        Returns:
          bool: success
        """
        if isinstance(cache_data, six.text_type):
            cache_data = cache_data.encode(encoding=self._encoding)
        with self._thread_lock:
            if not self._EnsureFileExists():
                return False
            with self._process_lock_getter() as acquired_plock:
                if not acquired_plock:
                    return False
                with open(self._filename, 'wb') as f:
                    f.write(cache_data)
                return True

    def _EnsureFileExists(self):
        """Touches a file; returns False on error, True on success."""
        if not os.path.exists(self._filename):
            old_umask = os.umask(127)
            try:
                open(self._filename, 'a+b').close()
            except OSError:
                return False
            finally:
                os.umask(old_umask)
        return True