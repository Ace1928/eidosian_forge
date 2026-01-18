import contextlib
import errno
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple
from . import debug, errors, osutils, trace
from .hooks import Hooks
from .i18n import gettext
from .transport import Transport
class _fcntl_ReadLock(_fcntl_FileLock):
    _open_locks: Dict[str, int] = {}

    def __init__(self, filename):
        super().__init__()
        self.filename = osutils.realpath(filename)
        if self.filename in _fcntl_WriteLock._open_locks:
            if 'strict_locks' in debug.debug_flags:
                raise errors.LockContention(self.filename)
            else:
                trace.mutter('Read lock taken w/ an open write lock on: %s' % (self.filename,))
        _fcntl_ReadLock._open_locks.setdefault(self.filename, 0)
        _fcntl_ReadLock._open_locks[self.filename] += 1
        self._open(filename, 'rb')
        try:
            fcntl.lockf(self.f, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as e:
            raise errors.LockContention(self.filename, e)

    def unlock(self):
        count = _fcntl_ReadLock._open_locks[self.filename]
        if count == 1:
            del _fcntl_ReadLock._open_locks[self.filename]
        else:
            _fcntl_ReadLock._open_locks[self.filename] = count - 1
        self._unlock()

    def temporary_write_lock(self):
        """Try to grab a write lock on the file.

            On platforms that support it, this will upgrade to a write lock
            without unlocking the file.
            Otherwise, this will release the read lock, and try to acquire a
            write lock.

            :return: A token which can be used to switch back to a read lock.
            """
        if self.filename in _fcntl_WriteLock._open_locks:
            raise AssertionError('file already locked: %r' % (self.filename,))
        try:
            wlock = _fcntl_TemporaryWriteLock(self)
        except errors.LockError:
            return (False, self)
        return (True, wlock)