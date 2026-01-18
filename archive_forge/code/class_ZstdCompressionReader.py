from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
class ZstdCompressionReader(object):
    """Readable compressing stream wrapper.

    ``ZstdCompressionReader`` is a read-only stream interface for obtaining
    compressed data from a source.

    This type conforms to the ``io.RawIOBase`` interface and should be usable
    by any type that operates against a *file-object* (``typing.BinaryIO``
    in Python type hinting speak).

    Instances are neither writable nor seekable (even if the underlying
    source is seekable). ``readline()`` and ``readlines()`` are not implemented
    because they don't make sense for compressed data. ``tell()`` returns the
    number of compressed bytes emitted so far.

    Instances are obtained by calling :py:meth:`ZstdCompressor.stream_reader`.

    In this example, we open a file for reading and then wrap that file
    handle with a stream from which compressed data can be ``read()``.

    >>> with open(path, 'rb') as fh:
    ...     cctx = zstandard.ZstdCompressor()
    ...     reader = cctx.stream_reader(fh)
    ...     while True:
    ...         chunk = reader.read(16384)
    ...         if not chunk:
    ...             break
    ...
    ...         # Do something with compressed chunk.

    Instances can also be used as context managers:

    >>> with open(path, 'rb') as fh:
    ...     cctx = zstandard.ZstdCompressor()
    ...     with cctx.stream_reader(fh) as reader:
    ...         while True:
    ...             chunk = reader.read(16384)
    ...             if not chunk:
    ...                 break
    ...
    ...             # Do something with compressed chunk.

    When the context manager exits or ``close()`` is called, the stream is
    closed, underlying resources are released, and future operations against
    the compression stream will fail.

    ``stream_reader()`` accepts a ``size`` argument specifying how large the
    input stream is. This is used to adjust compression parameters so they are
    tailored to the source size. e.g.

    >>> with open(path, 'rb') as fh:
    ...     cctx = zstandard.ZstdCompressor()
    ...     with cctx.stream_reader(fh, size=os.stat(path).st_size) as reader:
    ...         ...

    If the ``source`` is a stream, you can specify how large ``read()``
    requests to that stream should be via the ``read_size`` argument.
    It defaults to ``zstandard.COMPRESSION_RECOMMENDED_INPUT_SIZE``. e.g.

    >>> with open(path, 'rb') as fh:
    ...     cctx = zstandard.ZstdCompressor()
    ...     # Will perform fh.read(8192) when obtaining data to feed into the
    ...     # compressor.
    ...     with cctx.stream_reader(fh, read_size=8192) as reader:
    ...         ...
    """

    def __init__(self, compressor, source, read_size, closefd=True):
        self._compressor = compressor
        self._source = source
        self._read_size = read_size
        self._closefd = closefd
        self._entered = False
        self._closed = False
        self._bytes_compressed = 0
        self._finished_input = False
        self._finished_output = False
        self._in_buffer = ffi.new('ZSTD_inBuffer *')
        self._source_buffer = None

    def __enter__(self):
        if self._entered:
            raise ValueError('cannot __enter__ multiple times')
        if self._closed:
            raise ValueError('stream is closed')
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._entered = False
        self._compressor = None
        self.close()
        self._source = None
        return False

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return False

    def readline(self):
        raise io.UnsupportedOperation()

    def readlines(self):
        raise io.UnsupportedOperation()

    def write(self, data):
        raise OSError('stream is not writable')

    def writelines(self, ignored):
        raise OSError('stream is not writable')

    def isatty(self):
        return False

    def flush(self):
        return None

    def close(self):
        if self._closed:
            return
        self._closed = True
        f = getattr(self._source, 'close', None)
        if self._closefd and f:
            f()

    @property
    def closed(self):
        return self._closed

    def tell(self):
        return self._bytes_compressed

    def readall(self):
        chunks = []
        while True:
            chunk = self.read(1048576)
            if not chunk:
                break
            chunks.append(chunk)
        return b''.join(chunks)

    def __iter__(self):
        raise io.UnsupportedOperation()

    def __next__(self):
        raise io.UnsupportedOperation()
    next = __next__

    def _read_input(self):
        if self._finished_input:
            return
        if hasattr(self._source, 'read'):
            data = self._source.read(self._read_size)
            if not data:
                self._finished_input = True
                return
            self._source_buffer = ffi.from_buffer(data)
            self._in_buffer.src = self._source_buffer
            self._in_buffer.size = len(self._source_buffer)
            self._in_buffer.pos = 0
        else:
            self._source_buffer = ffi.from_buffer(self._source)
            self._in_buffer.src = self._source_buffer
            self._in_buffer.size = len(self._source_buffer)
            self._in_buffer.pos = 0

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

    def read(self, size=-1):
        if self._closed:
            raise ValueError('stream is closed')
        if size < -1:
            raise ValueError('cannot read negative amounts less than -1')
        if size == -1:
            return self.readall()
        if self._finished_output or size == 0:
            return b''
        dst_buffer = ffi.new('char[]', size)
        out_buffer = ffi.new('ZSTD_outBuffer *')
        out_buffer.dst = dst_buffer
        out_buffer.size = size
        out_buffer.pos = 0
        if self._compress_into_buffer(out_buffer):
            return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
        while not self._finished_input:
            self._read_input()
            if self._compress_into_buffer(out_buffer):
                return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
        old_pos = out_buffer.pos
        zresult = lib.ZSTD_compressStream2(self._compressor._cctx, out_buffer, self._in_buffer, lib.ZSTD_e_end)
        self._bytes_compressed += out_buffer.pos - old_pos
        if lib.ZSTD_isError(zresult):
            raise ZstdError('error ending compression stream: %s', _zstd_error(zresult))
        if zresult == 0:
            self._finished_output = True
        return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

    def read1(self, size=-1):
        if self._closed:
            raise ValueError('stream is closed')
        if size < -1:
            raise ValueError('cannot read negative amounts less than -1')
        if self._finished_output or size == 0:
            return b''
        if size == -1:
            size = COMPRESSION_RECOMMENDED_OUTPUT_SIZE
        dst_buffer = ffi.new('char[]', size)
        out_buffer = ffi.new('ZSTD_outBuffer *')
        out_buffer.dst = dst_buffer
        out_buffer.size = size
        out_buffer.pos = 0
        self._compress_into_buffer(out_buffer)
        if out_buffer.pos:
            return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
        while not self._finished_input:
            self._read_input()
            if self._compress_into_buffer(out_buffer):
                return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
            if out_buffer.pos and (not self._finished_input):
                return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
        old_pos = out_buffer.pos
        zresult = lib.ZSTD_compressStream2(self._compressor._cctx, out_buffer, self._in_buffer, lib.ZSTD_e_end)
        self._bytes_compressed += out_buffer.pos - old_pos
        if lib.ZSTD_isError(zresult):
            raise ZstdError('error ending compression stream: %s' % _zstd_error(zresult))
        if zresult == 0:
            self._finished_output = True
        return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

    def readinto(self, b):
        if self._closed:
            raise ValueError('stream is closed')
        if self._finished_output:
            return 0
        dest_buffer = ffi.from_buffer(b)
        ffi.memmove(b, b'', 0)
        out_buffer = ffi.new('ZSTD_outBuffer *')
        out_buffer.dst = dest_buffer
        out_buffer.size = len(dest_buffer)
        out_buffer.pos = 0
        if self._compress_into_buffer(out_buffer):
            return out_buffer.pos
        while not self._finished_input:
            self._read_input()
            if self._compress_into_buffer(out_buffer):
                return out_buffer.pos
        old_pos = out_buffer.pos
        zresult = lib.ZSTD_compressStream2(self._compressor._cctx, out_buffer, self._in_buffer, lib.ZSTD_e_end)
        self._bytes_compressed += out_buffer.pos - old_pos
        if lib.ZSTD_isError(zresult):
            raise ZstdError('error ending compression stream: %s', _zstd_error(zresult))
        if zresult == 0:
            self._finished_output = True
        return out_buffer.pos

    def readinto1(self, b):
        if self._closed:
            raise ValueError('stream is closed')
        if self._finished_output:
            return 0
        dest_buffer = ffi.from_buffer(b)
        ffi.memmove(b, b'', 0)
        out_buffer = ffi.new('ZSTD_outBuffer *')
        out_buffer.dst = dest_buffer
        out_buffer.size = len(dest_buffer)
        out_buffer.pos = 0
        self._compress_into_buffer(out_buffer)
        if out_buffer.pos:
            return out_buffer.pos
        while not self._finished_input:
            self._read_input()
            if self._compress_into_buffer(out_buffer):
                return out_buffer.pos
            if out_buffer.pos and (not self._finished_input):
                return out_buffer.pos
        old_pos = out_buffer.pos
        zresult = lib.ZSTD_compressStream2(self._compressor._cctx, out_buffer, self._in_buffer, lib.ZSTD_e_end)
        self._bytes_compressed += out_buffer.pos - old_pos
        if lib.ZSTD_isError(zresult):
            raise ZstdError('error ending compression stream: %s' % _zstd_error(zresult))
        if zresult == 0:
            self._finished_output = True
        return out_buffer.pos