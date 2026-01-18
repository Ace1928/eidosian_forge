import itertools
import os
import struct
from copy import copy
from io import BytesIO
from typing import Any, Tuple
from zlib import adler32
from ..lazy_import import lazy_import
import fastbencode as bencode
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import graph as _mod_graph
from .. import osutils
from .. import transport as _mod_transport
from ..registry import Registry
from ..textmerge import TextMerge
from . import index
class FileContentFactory(ContentFactory):
    """File-based content factory.
    """

    def __init__(self, key, parents, fileobj, sha1=None, size=None):
        self.key = key
        self.parents = parents
        self.file = fileobj
        self.storage_kind = 'file'
        self.sha1 = sha1
        self.size = size
        self._needs_reset = False

    def get_bytes_as(self, storage_kind):
        if self._needs_reset:
            self.file.seek(0)
        self._needs_reset = True
        if storage_kind == 'fulltext':
            return self.file.read()
        elif storage_kind == 'chunked':
            return list(osutils.file_iterator(self.file))
        elif storage_kind == 'lines':
            return list(self.file.readlines())
        raise UnavailableRepresentation(self.key, storage_kind, self.storage_kind)

    def iter_bytes_as(self, storage_kind):
        if self._needs_reset:
            self.file.seek(0)
        self._needs_reset = True
        if storage_kind == 'chunked':
            return osutils.file_iterator(self.file)
        elif storage_kind == 'lines':
            return self.file
        raise UnavailableRepresentation(self.key, storage_kind, self.storage_kind)