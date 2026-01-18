import struct, sys, time, os
import zlib
import builtins
import io
import _compression
class _GzipReader(_compression.DecompressReader):

    def __init__(self, fp):
        super().__init__(_PaddedFile(fp), zlib.decompressobj, wbits=-zlib.MAX_WBITS)
        self._new_member = True
        self._last_mtime = None

    def _init_read(self):
        self._crc = zlib.crc32(b'')
        self._stream_size = 0

    def _read_gzip_header(self):
        last_mtime = _read_gzip_header(self._fp)
        if last_mtime is None:
            return False
        self._last_mtime = last_mtime
        return True

    def read(self, size=-1):
        if size < 0:
            return self.readall()
        if not size:
            return b''
        while True:
            if self._decompressor.eof:
                self._read_eof()
                self._new_member = True
                self._decompressor = self._decomp_factory(**self._decomp_args)
            if self._new_member:
                self._init_read()
                if not self._read_gzip_header():
                    self._size = self._pos
                    return b''
                self._new_member = False
            buf = self._fp.read(io.DEFAULT_BUFFER_SIZE)
            uncompress = self._decompressor.decompress(buf, size)
            if self._decompressor.unconsumed_tail != b'':
                self._fp.prepend(self._decompressor.unconsumed_tail)
            elif self._decompressor.unused_data != b'':
                self._fp.prepend(self._decompressor.unused_data)
            if uncompress != b'':
                break
            if buf == b'':
                raise EOFError('Compressed file ended before the end-of-stream marker was reached')
        self._add_read_data(uncompress)
        self._pos += len(uncompress)
        return uncompress

    def _add_read_data(self, data):
        self._crc = zlib.crc32(data, self._crc)
        self._stream_size = self._stream_size + len(data)

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

    def _rewind(self):
        super()._rewind()
        self._new_member = True