import zlib
from gitdb.exc import (
from gitdb.util import (
from gitdb.fun import (
from gitdb.base import (      # Amazing !
from gitdb.stream import (
from struct import pack
from binascii import crc32
from gitdb.const import NULL_BYTE
import tempfile
import array
import os
import sys
def _iter_objects(self, as_stream):
    """Iterate over all objects in our index and yield their OInfo or OStream instences"""
    _sha = self._index.sha
    _object = self._object
    for index in range(self._index.size()):
        yield _object(_sha(index), as_stream, index)