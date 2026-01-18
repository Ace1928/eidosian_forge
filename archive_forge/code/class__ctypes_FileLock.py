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
class _ctypes_FileLock(_OSLock):

    def _open(self, filename, access, share, cflags, pymode):
        self.filename = osutils.realpath(filename)
        handle = _CreateFile(filename, access, share, None, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0)
        if handle in (INVALID_HANDLE_VALUE, 0):
            e = ctypes.WinError()
            if e.args[0] == ERROR_ACCESS_DENIED:
                raise errors.LockFailed(filename, e)
            if e.args[0] == ERROR_SHARING_VIOLATION:
                raise errors.LockContention(filename, e)
            raise e
        fd = msvcrt.open_osfhandle(handle, cflags)
        self.f = os.fdopen(fd, pymode)
        return self.f

    def unlock(self):
        self._clear_f()