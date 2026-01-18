from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def _decompress_into_buffer(self, out_buffer):
    """Decompress available input into an output buffer.

        Returns True if data in output buffer should be emitted.
        """
    zresult = lib.ZSTD_decompressStream(self._decompressor._dctx, out_buffer, self._in_buffer)
    if self._in_buffer.pos == self._in_buffer.size:
        self._in_buffer.src = ffi.NULL
        self._in_buffer.pos = 0
        self._in_buffer.size = 0
        self._source_buffer = None
        if not hasattr(self._source, 'read'):
            self._finished_input = True
    if lib.ZSTD_isError(zresult):
        raise ZstdError('zstd decompress error: %s' % _zstd_error(zresult))
    return out_buffer.pos and (out_buffer.pos == out_buffer.size or (zresult == 0 and (not self._read_across_frames)))