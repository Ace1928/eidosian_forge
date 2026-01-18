import os
import sys
import stat
import fnmatch
import collections
import errno
def _rmtree_isdir(entry):
    try:
        return entry.is_dir(follow_symlinks=False)
    except OSError:
        return False