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
def _sha_v1(self, i):
    """see ``_sha_v2``"""
    base = 1024 + i * 24 + 4
    return self._cursor.map()[base:base + 20]