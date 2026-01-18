import errno
import threading
from time import sleep
import weakref
class _LockFile(object):

    def __init__(self, path):
        self._path = path
        self._file = None

    def acquire(self, timeout=None, retry_period=None):
        fileobj = open(self._path, 'wb')
        try:
            if timeout is None and _lock_file_blocking_available:
                _lock_file_blocking(fileobj)
            else:
                _acquire_non_blocking(acquire=lambda: _lock_file_non_blocking(fileobj), timeout=timeout, retry_period=retry_period, path=self._path)
        except:
            fileobj.close()
            raise
        else:
            self._file = fileobj

    def release(self):
        if self._file is None:
            raise LockError('cannot release unlocked lock')
        _unlock_file(self._file)
        self._file.close()
        self._file = None