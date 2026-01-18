from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def _compress_into_buffer(self, out_buffer):
    if self._in_buffer.pos >= self._in_buffer.size:
        return
    old_pos = out_buffer.pos
    zresult = lib.ZSTD_compressStream2(self._compressor._cctx, out_buffer, self._in_buffer, lib.ZSTD_e_continue)
    self._bytes_compressed += out_buffer.pos - old_pos
    if self._in_buffer.pos == self._in_buffer.size:
        self._in_buffer.src = ffi.NULL
        self._in_buffer.pos = 0
        self._in_buffer.size = 0
        self._source_buffer = None
        if not hasattr(self._source, 'read'):
            self._finished_input = True
    if lib.ZSTD_isError(zresult):
        raise ZstdError('zstd compress error: %s', _zstd_error(zresult))
    return out_buffer.pos and out_buffer.pos == out_buffer.size