from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
class ZstdCompressionObj(object):
    """A compressor conforming to the API in Python's standard library.

    This type implements an API similar to compression types in Python's
    standard library such as ``zlib.compressobj`` and ``bz2.BZ2Compressor``.
    This enables existing code targeting the standard library API to swap
    in this type to achieve zstd compression.

    .. important::

       The design of this API is not ideal for optimal performance.

       The reason performance is not optimal is because the API is limited to
       returning a single buffer holding compressed data. When compressing
       data, we don't know how much data will be emitted. So in order to
       capture all this data in a single buffer, we need to perform buffer
       reallocations and/or extra memory copies. This can add significant
       overhead depending on the size or nature of the compressed data how
       much your application calls this type.

       If performance is critical, consider an API like
       :py:meth:`ZstdCompressor.stream_reader`,
       :py:meth:`ZstdCompressor.stream_writer`,
       :py:meth:`ZstdCompressor.chunker`, or
       :py:meth:`ZstdCompressor.read_to_iter`, which result in less overhead
       managing buffers.

    Instances are obtained by calling :py:meth:`ZstdCompressor.compressobj`.

    Here is how this API should be used:

    >>> cctx = zstandard.ZstdCompressor()
    >>> cobj = cctx.compressobj()
    >>> data = cobj.compress(b"raw input 0")
    >>> data = cobj.compress(b"raw input 1")
    >>> data = cobj.flush()

    Or to flush blocks:

    >>> cctx.zstandard.ZstdCompressor()
    >>> cobj = cctx.compressobj()
    >>> data = cobj.compress(b"chunk in first block")
    >>> data = cobj.flush(zstandard.COMPRESSOBJ_FLUSH_BLOCK)
    >>> data = cobj.compress(b"chunk in second block")
    >>> data = cobj.flush()

    For best performance results, keep input chunks under 256KB. This avoids
    extra allocations for a large output object.

    It is possible to declare the input size of the data that will be fed
    into the compressor:

    >>> cctx = zstandard.ZstdCompressor()
    >>> cobj = cctx.compressobj(size=6)
    >>> data = cobj.compress(b"foobar")
    >>> data = cobj.flush()
    """

    def compress(self, data):
        """Send data to the compressor.

        This method receives bytes to feed to the compressor and returns
        bytes constituting zstd compressed data.

        The zstd compressor accumulates bytes and the returned bytes may be
        substantially smaller or larger than the size of the input data on
        any given call. The returned value may be the empty byte string
        (``b""``).

        :param data:
           Data to write to the compressor.
        :return:
           Compressed data.
        """
        if self._finished:
            raise ZstdError('cannot call compress() after compressor finished')
        data_buffer = ffi.from_buffer(data)
        source = ffi.new('ZSTD_inBuffer *')
        source.src = data_buffer
        source.size = len(data_buffer)
        source.pos = 0
        chunks = []
        while source.pos < len(data):
            zresult = lib.ZSTD_compressStream2(self._compressor._cctx, self._out, source, lib.ZSTD_e_continue)
            if lib.ZSTD_isError(zresult):
                raise ZstdError('zstd compress error: %s' % _zstd_error(zresult))
            if self._out.pos:
                chunks.append(ffi.buffer(self._out.dst, self._out.pos)[:])
                self._out.pos = 0
        return b''.join(chunks)

    def flush(self, flush_mode=COMPRESSOBJ_FLUSH_FINISH):
        """Emit data accumulated in the compressor that hasn't been outputted yet.

        The ``flush_mode`` argument controls how to end the stream.

        ``zstandard.COMPRESSOBJ_FLUSH_FINISH`` (the default) ends the
        compression stream and finishes a zstd frame. Once this type of flush
        is performed, ``compress()`` and ``flush()`` can no longer be called.
        This type of flush **must** be called to end the compression context. If
        not called, the emitted data may be incomplete and may not be readable
        by a decompressor.

        ``zstandard.COMPRESSOBJ_FLUSH_BLOCK`` will flush a zstd block. This
        ensures that all data fed to this instance will have been omitted and
        can be decoded by a decompressor. Flushes of this type can be performed
        multiple times. The next call to ``compress()`` will begin a new zstd
        block.

        :param flush_mode:
           How to flush the zstd compressor.
        :return:
           Compressed data.
        """
        if flush_mode not in (COMPRESSOBJ_FLUSH_FINISH, COMPRESSOBJ_FLUSH_BLOCK):
            raise ValueError('flush mode not recognized')
        if self._finished:
            raise ZstdError('compressor object already finished')
        if flush_mode == COMPRESSOBJ_FLUSH_BLOCK:
            z_flush_mode = lib.ZSTD_e_flush
        elif flush_mode == COMPRESSOBJ_FLUSH_FINISH:
            z_flush_mode = lib.ZSTD_e_end
            self._finished = True
        else:
            raise ZstdError('unhandled flush mode')
        assert self._out.pos == 0
        in_buffer = ffi.new('ZSTD_inBuffer *')
        in_buffer.src = ffi.NULL
        in_buffer.size = 0
        in_buffer.pos = 0
        chunks = []
        while True:
            zresult = lib.ZSTD_compressStream2(self._compressor._cctx, self._out, in_buffer, z_flush_mode)
            if lib.ZSTD_isError(zresult):
                raise ZstdError('error ending compression stream: %s' % _zstd_error(zresult))
            if self._out.pos:
                chunks.append(ffi.buffer(self._out.dst, self._out.pos)[:])
                self._out.pos = 0
            if not zresult:
                break
        return b''.join(chunks)