import lz4
import io
import os
import builtins
import sys
from ._frame import (  # noqa: F401
class LZ4FrameFile(_compression.BaseStream):
    """A file object providing transparent LZ4F (de)compression.

    An LZ4FFile can act as a wrapper for an existing file object, or refer
    directly to a named file on disk.

    Note that LZ4FFile provides a *binary* file interface - data read is
    returned as bytes, and data to be written must be given as bytes.

    When opening a file for writing, the settings used by the compressor can be
    specified. The underlying compressor object is
    `lz4.frame.LZ4FrameCompressor`. See the docstrings for that class for
    details on compression options.

    Args:
        filename(str, bytes, PathLike, file object): can be either an actual
            file name (given as a str, bytes, or
            PathLike object), in which case the named file is opened, or it
            can be an existing file object to read from or write to.

    Keyword Args:
        mode(str): mode can be ``'r'`` for reading (default), ``'w'`` for
            (over)writing, ``'x'`` for creating exclusively, or ``'a'``
            for appending. These can equivalently be given as ``'rb'``,
            ``'wb'``, ``'xb'`` and ``'ab'`` respectively.
        return_bytearray (bool): When ``False`` a bytes object is returned from
            the calls to methods of this class. When ``True`` a ``bytearray``
            object will be returned. The default is ``False``.
        source_size (int): Optionally specify the total size of the
            uncompressed data. If specified, will be stored in the compressed
            frame header as an 8-byte field for later use during decompression.
            Default is ``0`` (no size stored). Only used for writing
            compressed files.
        block_size (int): Compressor setting. See
            `lz4.frame.LZ4FrameCompressor`.
        block_linked (bool): Compressor setting. See
            `lz4.frame.LZ4FrameCompressor`.
        compression_level (int): Compressor setting. See
            `lz4.frame.LZ4FrameCompressor`.
        content_checksum (bool): Compressor setting. See
            `lz4.frame.LZ4FrameCompressor`.
        block_checksum (bool): Compressor setting. See
            `lz4.frame.LZ4FrameCompressor`.
        auto_flush (bool): Compressor setting. See
            `lz4.frame.LZ4FrameCompressor`.

    """

    def __init__(self, filename=None, mode='r', block_size=BLOCKSIZE_DEFAULT, block_linked=True, compression_level=COMPRESSIONLEVEL_MIN, content_checksum=False, block_checksum=False, auto_flush=False, return_bytearray=False, source_size=0):
        self._fp = None
        self._closefp = False
        self._mode = _MODE_CLOSED
        if mode in ('r', 'rb'):
            mode_code = _MODE_READ
        elif mode in ('w', 'wb', 'a', 'ab', 'x', 'xb'):
            mode_code = _MODE_WRITE
            self._compressor = LZ4FrameCompressor(block_size=block_size, block_linked=block_linked, compression_level=compression_level, content_checksum=content_checksum, block_checksum=block_checksum, auto_flush=auto_flush, return_bytearray=return_bytearray)
            self._pos = 0
        else:
            raise ValueError('Invalid mode: {!r}'.format(mode))
        if sys.version_info > (3, 6):
            path_test = isinstance(filename, (str, bytes, os.PathLike))
        else:
            path_test = isinstance(filename, (str, bytes))
        if path_test is True:
            if 'b' not in mode:
                mode += 'b'
            self._fp = builtins.open(filename, mode)
            self._closefp = True
            self._mode = mode_code
        elif hasattr(filename, 'read') or hasattr(filename, 'write'):
            self._fp = filename
            self._mode = mode_code
        else:
            raise TypeError('filename must be a str, bytes, file or PathLike object')
        if self._mode == _MODE_READ:
            raw = _compression.DecompressReader(self._fp, LZ4FrameDecompressor)
            self._buffer = io.BufferedReader(raw)
        if self._mode == _MODE_WRITE:
            self._source_size = source_size
            self._fp.write(self._compressor.begin(source_size=source_size))

    def close(self):
        """Flush and close the file.

        May be called more than once without error. Once the file is
        closed, any other operation on it will raise a ValueError.
        """
        if self._mode == _MODE_CLOSED:
            return
        try:
            if self._mode == _MODE_READ:
                self._buffer.close()
                self._buffer = None
            elif self._mode == _MODE_WRITE:
                self.flush()
                self._compressor = None
        finally:
            try:
                if self._closefp:
                    self._fp.close()
            finally:
                self._fp = None
                self._closefp = False
                self._mode = _MODE_CLOSED

    @property
    def closed(self):
        """Returns ``True`` if this file is closed.

        Returns:
            bool: ``True`` if the file is closed, ``False`` otherwise.

        """
        return self._mode == _MODE_CLOSED

    def fileno(self):
        """Return the file descriptor for the underlying file.

        Returns:
            file object: file descriptor for file.

        """
        self._check_not_closed()
        return self._fp.fileno()

    def seekable(self):
        """Return whether the file supports seeking.

        Returns:
            bool: ``True`` if the file supports seeking, ``False`` otherwise.

        """
        return self.readable() and self._buffer.seekable()

    def readable(self):
        """Return whether the file was opened for reading.

        Returns:
            bool: ``True`` if the file was opened for reading, ``False``
                otherwise.

        """
        self._check_not_closed()
        return self._mode == _MODE_READ

    def writable(self):
        """Return whether the file was opened for writing.

        Returns:
            bool: ``True`` if the file was opened for writing, ``False``
                otherwise.

        """
        self._check_not_closed()
        return self._mode == _MODE_WRITE

    def peek(self, size=-1):
        """Return buffered data without advancing the file position.

        Always returns at least one byte of data, unless at EOF. The exact
        number of bytes returned is unspecified.

        Returns:
            bytes: uncompressed data

        """
        self._check_can_read()
        return self._buffer.peek(size)

    def readall(self):
        chunks = bytearray()
        while True:
            data = self.read(io.DEFAULT_BUFFER_SIZE)
            chunks += data
            if not data:
                break
        return bytes(chunks)

    def read(self, size=-1):
        """Read up to ``size`` uncompressed bytes from the file.

        If ``size`` is negative or omitted, read until ``EOF`` is reached.
        Returns ``b''`` if the file is already at ``EOF``.

        Args:
            size(int): If non-negative, specifies the maximum number of
                uncompressed bytes to return.

        Returns:
            bytes: uncompressed data

        """
        self._check_can_read()
        if size < 0 and sys.version_info >= (3, 10):
            return self.readall()
        return self._buffer.read(size)

    def read1(self, size=-1):
        """Read up to ``size`` uncompressed bytes.

        This method tries to avoid making multiple reads from the underlying
        stream.

        This method reads up to a buffer's worth of data if ``size`` is
        negative.

        Returns ``b''`` if the file is at EOF.

        Args:
            size(int): If non-negative, specifies the maximum number of
                uncompressed bytes to return.

        Returns:
            bytes: uncompressed data

        """
        self._check_can_read()
        if size < 0:
            size = io.DEFAULT_BUFFER_SIZE
        return self._buffer.read1(size)

    def readline(self, size=-1):
        """Read a line of uncompressed bytes from the file.

        The terminating newline (if present) is retained. If size is
        non-negative, no more than size bytes will be read (in which case the
        line may be incomplete). Returns b'' if already at EOF.

        Args:
            size(int): If non-negative, specifies the maximum number of
                uncompressed bytes to return.

        Returns:
            bytes: uncompressed data

        """
        self._check_can_read()
        return self._buffer.readline(size)

    def write(self, data):
        """Write a bytes object to the file.

        Returns the number of uncompressed bytes written, which is
        always the length of data in bytes. Note that due to buffering,
        the file on disk may not reflect the data written until close()
        is called.

        Args:
            data(bytes): uncompressed data to compress and write to the file

        Returns:
            int: the number of uncompressed bytes written to the file

        """
        if isinstance(data, (bytes, bytearray)):
            length = len(data)
        else:
            data = memoryview(data)
            length = data.nbytes
        self._check_can_write()
        if not self._compressor.started():
            header = self._compressor.begin(source_size=self._source_size)
            self._fp.write(header)
        compressed = self._compressor.compress(data)
        self._fp.write(compressed)
        self._pos += length
        return length

    def flush(self):
        """Flush the file, keeping it open.

        May be called more than once without error. The file may continue
        to be used normally after flushing.
        """
        if self.writable() and self._compressor.has_context():
            self._fp.write(self._compressor.flush())
        self._fp.flush()

    def seek(self, offset, whence=io.SEEK_SET):
        """Change the file position.

        The new position is specified by ``offset``, relative to the position
        indicated by ``whence``. Possible values for ``whence`` are:

        - ``io.SEEK_SET`` or 0: start of stream (default): offset must not be
          negative
        - ``io.SEEK_CUR`` or 1: current stream position
        - ``io.SEEK_END`` or 2: end of stream; offset must not be positive

        Returns the new file position.

        Note that seeking is emulated, so depending on the parameters, this
        operation may be extremely slow.

        Args:
            offset(int): new position in the file
            whence(int): position with which ``offset`` is measured. Allowed
                values are 0, 1, 2. The default is 0 (start of stream).

        Returns:
            int: new file position

        """
        self._check_can_seek()
        return self._buffer.seek(offset, whence)

    def tell(self):
        """Return the current file position.

        Args:
            None

        Returns:
            int: file position

        """
        self._check_not_closed()
        if self._mode == _MODE_READ:
            return self._buffer.tell()
        return self._pos