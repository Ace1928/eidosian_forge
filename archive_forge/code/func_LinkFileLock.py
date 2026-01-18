from __future__ import absolute_import
import functools
import os
import socket
import threading
import warnings
def LinkFileLock(*args, **kwds):
    """Factory function provided for backwards compatibility.

    Do not use in new code.  Instead, import LinkLockFile from the
    lockfile.linklockfile module.
    """
    from . import linklockfile
    return _fl_helper(linklockfile.LinkLockFile, 'lockfile.linklockfile', *args, **kwds)