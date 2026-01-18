from __future__ import absolute_import
import functools
import os
import socket
import threading
import warnings
class _SharedBase(object):

    def __init__(self, path):
        self.path = path

    def acquire(self, timeout=None):
        """
        Acquire the lock.

        * If timeout is omitted (or None), wait forever trying to lock the
          file.

        * If timeout > 0, try to acquire the lock for that many seconds.  If
          the lock period expires and the file is still locked, raise
          LockTimeout.

        * If timeout <= 0, raise AlreadyLocked immediately if the file is
          already locked.
        """
        raise NotImplemented('implement in subclass')

    def release(self):
        """
        Release the lock.

        If the file is not locked, raise NotLocked.
        """
        raise NotImplemented('implement in subclass')

    def __enter__(self):
        """
        Context manager support.
        """
        self.acquire()
        return self

    def __exit__(self, *_exc):
        """
        Context manager support.
        """
        self.release()

    def __repr__(self):
        return '<%s: %r>' % (self.__class__.__name__, self.path)