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
def _entry_v2(self, i):
    """:return: tuple(offset, binsha, crc)"""
    return (self._offset_v2(i), self._sha_v2(i), self._crc_v2(i))