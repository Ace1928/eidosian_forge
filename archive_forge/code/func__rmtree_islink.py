import os
import sys
import stat
import fnmatch
import collections
import errno
def _rmtree_islink(path):
    return os.path.islink(path)