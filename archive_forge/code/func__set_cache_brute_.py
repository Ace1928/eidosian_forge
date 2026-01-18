from io import BytesIO
import mmap
import os
import sys
import zlib
from gitdb.fun import (
from gitdb.util import (
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_bytes
def _set_cache_brute_(self, attr):
    """If we are here, we apply the actual deltas"""
    buffer_info_list = list()
    max_target_size = 0
    for dstream in self._dstreams:
        buf = dstream.read(512)
        offset, src_size = msb_size(buf)
        offset, target_size = msb_size(buf, offset)
        buffer_info_list.append((buf[offset:], offset, src_size, target_size))
        max_target_size = max(max_target_size, target_size)
    base_size = self._bstream.size
    target_size = max_target_size
    if len(self._dstreams) > 1:
        base_size = target_size = max(base_size, max_target_size)
    bbuf = allocate_memory(base_size)
    stream_copy(self._bstream.read, bbuf.write, base_size, 256 * mmap.PAGESIZE)
    tbuf = allocate_memory(target_size)
    final_target_size = None
    for (dbuf, offset, src_size, target_size), dstream in zip(reversed(buffer_info_list), reversed(self._dstreams)):
        ddata = allocate_memory(dstream.size - offset)
        ddata.write(dbuf)
        stream_copy(dstream.read, ddata.write, dstream.size, 256 * mmap.PAGESIZE)
        if 'c_apply_delta' in globals():
            c_apply_delta(bbuf, ddata, tbuf)
        else:
            apply_delta_data(bbuf, src_size, ddata, len(ddata), tbuf.write)
        bbuf, tbuf = (tbuf, bbuf)
        bbuf.seek(0)
        tbuf.seek(0)
        final_target_size = target_size
    self._mm_target = bbuf
    self._size = final_target_size