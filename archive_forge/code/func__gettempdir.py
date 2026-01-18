import functools as _functools
import warnings as _warnings
import io as _io
import os as _os
import shutil as _shutil
import stat as _stat
import errno as _errno
from random import Random as _Random
import sys as _sys
import types as _types
import weakref as _weakref
import _thread
def _gettempdir():
    """Private accessor for tempfile.tempdir."""
    global tempdir
    if tempdir is None:
        _once_lock.acquire()
        try:
            if tempdir is None:
                tempdir = _get_default_tempdir()
        finally:
            _once_lock.release()
    return tempdir