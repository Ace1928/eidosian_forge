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
class _fcntl_WriteLock(_fcntl_FileLock):
    _open_locks: Set[str] = set()

    def __init__(self, filename):
        super().__init__()
        self.filename = osutils.realpath(filename)
        if self.filename in _fcntl_WriteLock._open_locks:
            self._clear_f()
            raise errors.LockContention(self.filename)
        if self.filename in _fcntl_ReadLock._open_locks:
            if 'strict_locks' in debug.debug_flags:
                self._clear_f()
                raise errors.LockContention(self.filename)
            else:
                trace.mutter('Write lock taken w/ an open read lock on: %s' % (self.filename,))
        self._open(self.filename, 'rb+')
        _fcntl_WriteLock._open_locks.add(self.filename)
        try:
            fcntl.lockf(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EACCES):
                self.unlock()
            raise errors.LockContention(self.filename, e)

    def unlock(self):
        _fcntl_WriteLock._open_locks.remove(self.filename)
        self._unlock()