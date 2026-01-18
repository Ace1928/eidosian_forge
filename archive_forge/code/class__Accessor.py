import fnmatch
import functools
import io
import ntpath
import os
import posixpath
import re
import sys
import time
from collections import Sequence
from contextlib import contextmanager
from errno import EINVAL, ENOENT
from operator import attrgetter
from stat import S_ISDIR, S_ISLNK, S_ISREG, S_ISSOCK, S_ISBLK, S_ISCHR, S_ISFIFO
class _Accessor:
    """An accessor implements a particular (system-specific or not) way of
    accessing paths on the filesystem."""