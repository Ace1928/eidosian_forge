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
def _iterate_directories(self, parent_path, is_dir, scandir):
    yield parent_path
    try:
        with scandir(parent_path) as scandir_it:
            entries = list(scandir_it)
        for entry in entries:
            entry_is_dir = False
            try:
                entry_is_dir = entry.is_dir(follow_symlinks=False)
            except OSError as e:
                if not _ignore_error(e):
                    raise
            if entry_is_dir:
                path = parent_path._make_child_relpath(entry.name)
                for p in self._iterate_directories(path, is_dir, scandir):
                    yield p
    except PermissionError:
        return