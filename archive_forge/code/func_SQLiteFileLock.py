from __future__ import absolute_import
import functools
import os
import socket
import threading
import warnings
def SQLiteFileLock(*args, **kwds):
    """Factory function provided for backwards compatibility.

    Do not use in new code.  Instead, import SQLiteLockFile from the
    lockfile.mkdirlockfile module.
    """
    from . import sqlitelockfile
    return _fl_helper(sqlitelockfile.SQLiteLockFile, 'lockfile.sqlitelockfile', *args, **kwds)