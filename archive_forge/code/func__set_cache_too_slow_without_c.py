from io import BytesIO
import mmap
import os
import sys
import zlib
from gitdb.fun import (
from gitdb.util import (
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_bytes
def _set_cache_too_slow_without_c(self, attr):
    if len(self._dstreams) == 1:
        return self._set_cache_brute_(attr)
    dcl = connect_deltas(self._dstreams)
    if dcl.rbound() == 0:
        self._size = 0
        self._mm_target = allocate_memory(0)
        return
    self._size = dcl.rbound()
    self._mm_target = allocate_memory(self._size)
    bbuf = allocate_memory(self._bstream.size)
    stream_copy(self._bstream.read, bbuf.write, self._bstream.size, 256 * mmap.PAGESIZE)
    write = self._mm_target.write
    dcl.apply(bbuf, write)
    self._mm_target.seek(0)