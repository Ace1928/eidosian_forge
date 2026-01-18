import os
import sys
import stat
import fnmatch
import collections
import errno
def _access_check(fn, mode):
    return os.path.exists(fn) and os.access(fn, mode) and (not os.path.isdir(fn))