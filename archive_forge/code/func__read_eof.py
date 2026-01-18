import struct, sys, time, os
import zlib
import builtins
import io
import _compression
def _read_eof(self):
    crc32, isize = struct.unpack('<II', _read_exact(self._fp, 8))
    if crc32 != self._crc:
        raise BadGzipFile('CRC check failed %s != %s' % (hex(crc32), hex(self._crc)))
    elif isize != self._stream_size & 4294967295:
        raise BadGzipFile('Incorrect length of data produced')
    c = b'\x00'
    while c == b'\x00':
        c = self._fp.read(1)
    if c:
        self._fp.prepend(c)