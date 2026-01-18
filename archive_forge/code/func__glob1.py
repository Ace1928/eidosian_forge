import contextlib
import os
import re
import fnmatch
import itertools
import stat
import sys
def _glob1(dirname, pattern, dir_fd, dironly, include_hidden=False):
    names = _listdir(dirname, dir_fd, dironly)
    if include_hidden or not _ishidden(pattern):
        names = (x for x in names if include_hidden or not _ishidden(x))
    return fnmatch.filter(names, pattern)