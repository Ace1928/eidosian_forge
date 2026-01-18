import fnmatch
import functools
import io
import ntpath
import os
import posixpath
import re
import sys
import warnings
from _collections_abc import Sequence
from errno import ENOENT, ENOTDIR, EBADF, ELOOP
from operator import attrgetter
from stat import S_ISDIR, S_ISLNK, S_ISREG, S_ISSOCK, S_ISBLK, S_ISCHR, S_ISFIFO
from urllib.parse import quote_from_bytes as urlquote_from_bytes
class _PreciseSelector(_Selector):

    def __init__(self, name, child_parts, flavour):
        self.name = name
        _Selector.__init__(self, child_parts, flavour)

    def _select_from(self, parent_path, is_dir, exists, scandir):
        try:
            path = parent_path._make_child_relpath(self.name)
            if (is_dir if self.dironly else exists)(path):
                for p in self.successor._select_from(path, is_dir, exists, scandir):
                    yield p
        except PermissionError:
            return