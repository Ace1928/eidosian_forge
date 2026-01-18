from __future__ import print_function, unicode_literals
import sys
import typing
import six
import zipfile
from datetime import datetime
from . import errors
from ._url_tools import url_quote
from .base import FS
from .compress import write_zip
from .enums import ResourceType, Seek
from .info import Info
from .iotools import RawWrapper
from .memoryfs import MemoryFS
from .opener import open_fs
from .path import dirname, forcedir, normpath, relpath
from .permissions import Permissions
from .time import datetime_to_epoch
from .wrapfs import WrapFS
def _path_to_zip_name(self, path):
    """Convert a path to a zip file name."""
    path = relpath(normpath(path))
    if self._directory.isdir(path):
        path = forcedir(path)
    if six.PY2:
        return path.encode(self.encoding)
    return path