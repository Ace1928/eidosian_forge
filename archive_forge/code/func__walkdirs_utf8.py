import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def _walkdirs_utf8(top, prefix='', fs_enc=None):
    """Yield data about all the directories in a tree.

    This yields the same information as walkdirs() only each entry is yielded
    in utf-8. On platforms which have a filesystem encoding of utf8 the paths
    are returned as exact byte-strings.

    :return: yields a tuple of (dir_info, [file_info])
        dir_info is (utf8_relpath, path-from-top)
        file_info is (utf8_relpath, utf8_name, kind, lstat, path-from-top)
        if top is an absolute path, path-from-top is also an absolute path.
        path-from-top might be unicode or utf8, but it is the correct path to
        pass to os functions to affect the file in question. (such as os.lstat)
    """
    global _selected_dir_reader
    if _selected_dir_reader is None:
        if fs_enc is None:
            fs_enc = sys.getfilesystemencoding()
        if sys.platform == 'win32':
            try:
                from ._walkdirs_win32 import Win32ReadDir
                _selected_dir_reader = Win32ReadDir()
            except ImportError:
                pass
        elif fs_enc in ('utf-8', 'ascii'):
            try:
                from ._readdir_pyx import UTF8DirReader
                _selected_dir_reader = UTF8DirReader()
            except ImportError as e:
                failed_to_load_extension(e)
                pass
    if _selected_dir_reader is None:
        _selected_dir_reader = UnicodeDirReader()
    pending = [[_selected_dir_reader.top_prefix_to_starting_dir(top, prefix)]]
    read_dir = _selected_dir_reader.read_dir
    _directory = _directory_kind
    while pending:
        relroot, _, _, _, top = pending[-1].pop()
        if not pending[-1]:
            pending.pop()
        dirblock = sorted(read_dir(relroot, top))
        yield ((relroot, top), dirblock)
        next = [d for d in reversed(dirblock) if d[2] == _directory]
        if next:
            pending.append(next)