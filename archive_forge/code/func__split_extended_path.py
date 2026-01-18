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
def _split_extended_path(self, s, ext_prefix=ext_namespace_prefix):
    prefix = ''
    if s.startswith(ext_prefix):
        prefix = s[:4]
        s = s[4:]
        if s.startswith('UNC\\'):
            prefix += s[:3]
            s = '\\' + s[3:]
    return (prefix, s)