from __future__ import print_function, unicode_literals
import typing
from typing import IO, cast
import os
import six
import tarfile
from collections import OrderedDict
from . import errors
from ._url_tools import url_quote
from .base import FS
from .compress import write_tar
from .enums import ResourceType
from .errors import IllegalBackReference, NoURL
from .info import Info
from .iotools import RawWrapper
from .opener import open_fs
from .path import basename, frombase, isbase, normpath, parts, relpath
from .permissions import Permissions
from .wrapfs import WrapFS
@property
def _directory_entries(self):
    """Lazy directory cache."""
    if self._directory_cache is None:
        _decode = self._decode
        _directory_entries = ((_decode(info.name).strip('/'), info) for info in self._tar)

        def _list_tar():
            for name, info in _directory_entries:
                try:
                    _name = normpath(name)
                except IllegalBackReference:
                    pass
                else:
                    if _name:
                        yield (_name, info)
        self._directory_cache = OrderedDict(_list_tar())
    return self._directory_cache