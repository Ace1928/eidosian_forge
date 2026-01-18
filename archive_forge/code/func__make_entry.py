import errno
import io
import os
import secrets
import shutil
from contextlib import suppress
from functools import cached_property, wraps
from urllib.parse import parse_qs
from fsspec.spec import AbstractFileSystem
from fsspec.utils import (
def _make_entry(self, info):
    from pyarrow.fs import FileType
    if info.type is FileType.Directory:
        kind = 'directory'
    elif info.type is FileType.File:
        kind = 'file'
    elif info.type is FileType.NotFound:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), info.path)
    else:
        kind = 'other'
    return {'name': info.path, 'size': info.size, 'type': kind, 'mtime': info.mtime}