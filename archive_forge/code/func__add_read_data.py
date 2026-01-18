import struct, sys, time, os
import zlib
import builtins
import io
import _compression
def _add_read_data(self, data):
    self._crc = zlib.crc32(data, self._crc)
    self._stream_size = self._stream_size + len(data)