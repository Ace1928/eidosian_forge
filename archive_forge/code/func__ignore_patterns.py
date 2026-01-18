import os
import sys
import stat
import fnmatch
import collections
import errno
def _ignore_patterns(path, names):
    ignored_names = []
    for pattern in patterns:
        ignored_names.extend(fnmatch.filter(names, pattern))
    return set(ignored_names)