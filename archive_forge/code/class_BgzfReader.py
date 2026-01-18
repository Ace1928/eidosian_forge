import struct
import sys
import zlib
from builtins import open as _open
class BgzfReader:
    """BGZF reader, acts like a read only handle but seek/tell differ.

    Let's use the BgzfBlocks function to have a peek at the BGZF blocks
    in an example BAM file,

    >>> from builtins import open
    >>> handle = open("SamBam/ex1.bam", "rb")
    >>> for values in BgzfBlocks(handle):
    ...     print("Raw start %i, raw length %i; data start %i, data length %i" % values)
    Raw start 0, raw length 18239; data start 0, data length 65536
    Raw start 18239, raw length 18223; data start 65536, data length 65536
    Raw start 36462, raw length 18017; data start 131072, data length 65536
    Raw start 54479, raw length 17342; data start 196608, data length 65536
    Raw start 71821, raw length 17715; data start 262144, data length 65536
    Raw start 89536, raw length 17728; data start 327680, data length 65536
    Raw start 107264, raw length 17292; data start 393216, data length 63398
    Raw start 124556, raw length 28; data start 456614, data length 0
    >>> handle.close()

    Now let's see how to use this block information to jump to
    specific parts of the decompressed BAM file:

    >>> handle = BgzfReader("SamBam/ex1.bam", "rb")
    >>> assert 0 == handle.tell()
    >>> magic = handle.read(4)
    >>> assert 4 == handle.tell()

    So far nothing so strange, we got the magic marker used at the
    start of a decompressed BAM file, and the handle position makes
    sense. Now however, let's jump to the end of this block and 4
    bytes into the next block by reading 65536 bytes,

    >>> data = handle.read(65536)
    >>> len(data)
    65536
    >>> assert 1195311108 == handle.tell()

    Expecting 4 + 65536 = 65540 were you? Well this is a BGZF 64-bit
    virtual offset, which means:

    >>> split_virtual_offset(1195311108)
    (18239, 4)

    You should spot 18239 as the start of the second BGZF block, while
    the 4 is the offset into this block. See also make_virtual_offset,

    >>> make_virtual_offset(18239, 4)
    1195311108

    Let's jump back to almost the start of the file,

    >>> make_virtual_offset(0, 2)
    2
    >>> handle.seek(2)
    2
    >>> handle.close()

    Note that you can use the max_cache argument to limit the number of
    BGZF blocks cached in memory. The default is 100, and since each
    block can be up to 64kb, the default cache could take up to 6MB of
    RAM. The cache is not important for reading through the file in one
    pass, but is important for improving performance of random access.
    """

    def __init__(self, filename=None, mode='r', fileobj=None, max_cache=100):
        """Initialize the class for reading a BGZF file.

        You would typically use the top level ``bgzf.open(...)`` function
        which will call this class internally. Direct use is discouraged.

        Either the ``filename`` (string) or ``fileobj`` (input file object in
        binary mode) arguments must be supplied, but not both.

        Argument ``mode`` controls if the data will be returned as strings in
        text mode ("rt", "tr", or default "r"), or bytes binary mode ("rb"
        or "br"). The argument name matches the built-in ``open(...)`` and
        standard library ``gzip.open(...)`` function.

        If text mode is requested, in order to avoid multi-byte characters,
        this is hard coded to use the "latin1" encoding, and "\\r" and "\\n"
        are passed as is (without implementing universal new line mode). There
        is no ``encoding`` argument.

        If your data is in UTF-8 or any other incompatible encoding, you must
        use binary mode, and decode the appropriate fragments yourself.

        Argument ``max_cache`` controls the maximum number of BGZF blocks to
        cache in memory. Each can be up to 64kb thus the default of 100 blocks
        could take up to 6MB of RAM. This is important for efficient random
        access, a small value is fine for reading the file in one pass.
        """
        if max_cache < 1:
            raise ValueError('Use max_cache with a minimum of 1')
        if filename and fileobj:
            raise ValueError('Supply either filename or fileobj, not both')
        if mode.lower() not in ('r', 'tr', 'rt', 'rb', 'br'):
            raise ValueError("Must use a read mode like 'r' (default), 'rt', or 'rb' for binary")
        if fileobj:
            if fileobj.read(0) != b'':
                raise ValueError('fileobj not opened in binary mode')
            handle = fileobj
        else:
            handle = _open(filename, 'rb')
        self._text = 'b' not in mode.lower()
        if self._text:
            self._newline = '\n'
        else:
            self._newline = b'\n'
        self._handle = handle
        self.max_cache = max_cache
        self._buffers = {}
        self._block_start_offset = None
        self._block_raw_length = None
        self._load_block(handle.tell())

    def _load_block(self, start_offset=None):
        if start_offset is None:
            start_offset = self._block_start_offset + self._block_raw_length
        if start_offset == self._block_start_offset:
            self._within_block_offset = 0
            return
        elif start_offset in self._buffers:
            self._buffer, self._block_raw_length = self._buffers[start_offset]
            self._within_block_offset = 0
            self._block_start_offset = start_offset
            return
        while len(self._buffers) >= self.max_cache:
            self._buffers.popitem()
        handle = self._handle
        if start_offset is not None:
            handle.seek(start_offset)
        self._block_start_offset = handle.tell()
        try:
            block_size, self._buffer = _load_bgzf_block(handle, self._text)
        except StopIteration:
            block_size = 0
            if self._text:
                self._buffer = ''
            else:
                self._buffer = b''
        self._within_block_offset = 0
        self._block_raw_length = block_size
        self._buffers[self._block_start_offset] = (self._buffer, block_size)

    def tell(self):
        """Return a 64-bit unsigned BGZF virtual offset."""
        if 0 < self._within_block_offset and self._within_block_offset == len(self._buffer):
            return self._block_start_offset + self._block_raw_length << 16
        else:
            return self._block_start_offset << 16 | self._within_block_offset

    def seek(self, virtual_offset):
        """Seek to a 64-bit unsigned BGZF virtual offset."""
        start_offset = virtual_offset >> 16
        within_block = virtual_offset ^ start_offset << 16
        if start_offset != self._block_start_offset:
            self._load_block(start_offset)
            if start_offset != self._block_start_offset:
                raise ValueError('start_offset not loaded correctly')
        if within_block > len(self._buffer):
            if not (within_block == 0 and len(self._buffer) == 0):
                raise ValueError('Within offset %i but block size only %i' % (within_block, len(self._buffer)))
        self._within_block_offset = within_block
        return virtual_offset

    def read(self, size=-1):
        """Read method for the BGZF module."""
        if size < 0:
            raise NotImplementedError("Don't be greedy, that could be massive!")
        result = '' if self._text else b''
        while size and self._block_raw_length:
            if self._within_block_offset + size <= len(self._buffer):
                data = self._buffer[self._within_block_offset:self._within_block_offset + size]
                self._within_block_offset += size
                if not data:
                    raise ValueError('Must be at least 1 byte')
                result += data
                break
            else:
                data = self._buffer[self._within_block_offset:]
                size -= len(data)
                self._load_block()
                result += data
        return result

    def readline(self):
        """Read a single line for the BGZF file."""
        result = '' if self._text else b''
        while self._block_raw_length:
            i = self._buffer.find(self._newline, self._within_block_offset)
            if i == -1:
                data = self._buffer[self._within_block_offset:]
                self._load_block()
                result += data
            elif i + 1 == len(self._buffer):
                data = self._buffer[self._within_block_offset:]
                self._load_block()
                if not data:
                    raise ValueError('Must be at least 1 byte')
                result += data
                break
            else:
                data = self._buffer[self._within_block_offset:i + 1]
                self._within_block_offset = i + 1
                result += data
                break
        return result

    def __next__(self):
        """Return the next line."""
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __iter__(self):
        """Iterate over the lines in the BGZF file."""
        return self

    def close(self):
        """Close BGZF file."""
        self._handle.close()
        self._buffer = None
        self._block_start_offset = None
        self._buffers = None

    def seekable(self):
        """Return True indicating the BGZF supports random access."""
        return True

    def isatty(self):
        """Return True if connected to a TTY device."""
        return False

    def fileno(self):
        """Return integer file descriptor."""
        return self._handle.fileno()

    def __enter__(self):
        """Open a file operable with WITH statement."""
        return self

    def __exit__(self, type, value, traceback):
        """Close a file with WITH statement."""
        self.close()