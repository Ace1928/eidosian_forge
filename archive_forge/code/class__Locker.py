import errno
import threading
from time import sleep
import weakref
class _Locker(object):
    """
    A lock wrapper to always apply the given *timeout* and *retry_period*
    to acquire() calls.
    """

    def __init__(self, lock, timeout=None, retry_period=None):
        self._lock = lock
        self._timeout = timeout
        self._retry_period = retry_period

    def acquire(self):
        self._lock.acquire(self._timeout, self._retry_period)

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()