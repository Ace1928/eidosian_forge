from __future__ import absolute_import
import functools
import os
import socket
import threading
import warnings
def _fl_helper(cls, mod, *args, **kwds):
    warnings.warn('Import from %s module instead of lockfile package' % mod, DeprecationWarning, stacklevel=2)
    if not isinstance(args[0], str):
        args = args[1:]
    if len(args) == 1 and (not kwds):
        kwds['threaded'] = True
    return cls(*args, **kwds)