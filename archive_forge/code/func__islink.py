import os
import sys
import stat
import fnmatch
import collections
import errno
def _islink(fn):
    return fn.is_symlink() if isinstance(fn, os.DirEntry) else os.path.islink(fn)