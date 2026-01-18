from __future__ import absolute_import
import functools
import os
import socket
import threading
import warnings
def MkdirFileLock(*args, **kwds):
    """Factory function provided for backwards compatibility.

    Do not use in new code.  Instead, import MkdirLockFile from the
    lockfile.mkdirlockfile module.
    """
    from . import mkdirlockfile
    return _fl_helper(mkdirlockfile.MkdirLockFile, 'lockfile.mkdirlockfile', *args, **kwds)