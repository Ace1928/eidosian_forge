from io import BytesIO
import mmap
import os
import sys
import zlib
from gitdb.fun import (
from gitdb.util import (
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_bytes
class DecompressMemMapReader(LazyMixin):
    """Reads data in chunks from a memory map and decompresses it. The client sees
    only the uncompressed data, respective file-like read calls are handling on-demand
    buffered decompression accordingly

    A constraint on the total size of bytes is activated, simulating
    a logical file within a possibly larger physical memory area

    To read efficiently, you clearly don't want to read individual bytes, instead,
    read a few kilobytes at least.

    **Note:** The chunk-size should be carefully selected as it will involve quite a bit
        of string copying due to the way the zlib is implemented. Its very wasteful,
        hence we try to find a good tradeoff between allocation time and number of
        times we actually allocate. An own zlib implementation would be good here
        to better support streamed reading - it would only need to keep the mmap
        and decompress it into chunks, that's all ... """
    __slots__ = ('_m', '_zip', '_buf', '_buflen', '_br', '_cws', '_cwe', '_s', '_close', '_cbr', '_phi')
    max_read_size = 512 * 1024

    def __init__(self, m, close_on_deletion, size=None):
        """Initialize with mmap for stream reading
        :param m: must be content data - use new if you have object data and no size"""
        self._m = m
        self._zip = zlib.decompressobj()
        self._buf = None
        self._buflen = 0
        if size is not None:
            self._s = size
        self._br = 0
        self._cws = 0
        self._cwe = 0
        self._cbr = 0
        self._phi = False
        self._close = close_on_deletion

    def _set_cache_(self, attr):
        assert attr == '_s'
        self._parse_header_info()

    def __del__(self):
        self.close()

    def _parse_header_info(self):
        """If this stream contains object data, parse the header info and skip the
        stream to a point where each read will yield object content

        :return: parsed type_string, size"""
        maxb = 8192
        self._s = maxb
        hdr = self.read(maxb)
        hdrend = hdr.find(NULL_BYTE)
        typ, size = hdr[:hdrend].split(BYTE_SPACE)
        size = int(size)
        self._s = size
        self._br = 0
        hdrend += 1
        self._buf = BytesIO(hdr[hdrend:])
        self._buflen = len(hdr) - hdrend
        self._phi = True
        return (typ, size)

    @classmethod
    def new(self, m, close_on_deletion=False):
        """Create a new DecompressMemMapReader instance for acting as a read-only stream
        This method parses the object header from m and returns the parsed
        type and size, as well as the created stream instance.

        :param m: memory map on which to operate. It must be object data ( header + contents )
        :param close_on_deletion: if True, the memory map will be closed once we are
            being deleted"""
        inst = DecompressMemMapReader(m, close_on_deletion, 0)
        typ, size = inst._parse_header_info()
        return (typ, size, inst)

    def data(self):
        """:return: random access compatible data we are working on"""
        return self._m

    def close(self):
        """Close our underlying stream of compressed bytes if this was allowed during initialization
        :return: True if we closed the underlying stream
        :note: can be called safely
        """
        if self._close:
            if hasattr(self._m, 'close'):
                self._m.close()
            self._close = False

    def compressed_bytes_read(self):
        """
        :return: number of compressed bytes read. This includes the bytes it
            took to decompress the header ( if there was one )"""
        if self._br == self._s and (not self._zip.unused_data):
            self._br = 0
            if hasattr(self._zip, 'status'):
                while self._zip.status == zlib.Z_OK:
                    self.read(mmap.PAGESIZE)
            else:
                while not self._zip.unused_data and self._cbr != len(self._m):
                    self.read(mmap.PAGESIZE)
            self._br = self._s
        return self._cbr

    def seek(self, offset, whence=getattr(os, 'SEEK_SET', 0)):
        """Allows to reset the stream to restart reading
        :raise ValueError: If offset and whence are not 0"""
        if offset != 0 or whence != getattr(os, 'SEEK_SET', 0):
            raise ValueError('Can only seek to position 0')
        self._zip = zlib.decompressobj()
        self._br = self._cws = self._cwe = self._cbr = 0
        if self._phi:
            self._phi = False
            del self._s

    def read(self, size=-1):
        if size < 1:
            size = self._s - self._br
        else:
            size = min(size, self._s - self._br)
        if size == 0:
            return b''
        dat = b''
        if self._buf:
            if self._buflen >= size:
                dat = self._buf.read(size)
                self._buflen -= size
                self._br += size
                return dat
            else:
                dat = self._buf.read()
                size -= self._buflen
                self._br += self._buflen
                self._buflen = 0
                self._buf = None
        tail = self._zip.unconsumed_tail
        if tail:
            self._cws = self._cwe - len(tail)
            self._cwe = self._cws + size
        else:
            cws = self._cws
            self._cws = self._cwe
            self._cwe = cws + size
        if self._cwe - self._cws < 8:
            self._cwe = self._cws + 8
        indata = self._m[self._cws:self._cwe]
        self._cwe = self._cws + len(indata)
        dcompdat = self._zip.decompress(indata, size)
        if getattr(zlib, 'ZLIB_RUNTIME_VERSION', zlib.ZLIB_VERSION) in ('1.2.7', '1.2.5') and (not sys.platform == 'darwin'):
            unused_datalen = len(self._zip.unconsumed_tail)
        else:
            unused_datalen = len(self._zip.unconsumed_tail) + len(self._zip.unused_data)
        self._cbr += len(indata) - unused_datalen
        self._br += len(dcompdat)
        if dat:
            dcompdat = dat + dcompdat
        if dcompdat and len(dcompdat) - len(dat) < size and (self._br < self._s):
            dcompdat += self.read(size - len(dcompdat))
        return dcompdat