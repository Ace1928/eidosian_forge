from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
class ZstdCompressor(object):
    """
    Create an object used to perform Zstandard compression.

    Each instance is essentially a wrapper around a ``ZSTD_CCtx`` from
    zstd's C API.

    An instance can compress data various ways. Instances can be used
    multiple times. Each compression operation will use the compression
    parameters defined at construction time.

    .. note:

       When using a compression dictionary and multiple compression
       operations are performed, the ``ZstdCompressionParameters`` derived
       from an integer compression ``level`` and the first compressed data's
       size will be reused for all subsequent operations. This may not be
       desirable if source data sizes vary significantly.

    ``compression_params`` is mutually exclusive with ``level``,
    ``write_checksum``, ``write_content_size``, ``write_dict_id``, and
    ``threads``.

    Assume that each ``ZstdCompressor`` instance can only handle a single
    logical compression operation at the same time. i.e. if you call a method
    like ``stream_reader()`` to obtain multiple objects derived from the same
    ``ZstdCompressor`` instance and attempt to use them simultaneously, errors
    will likely occur.

    If you need to perform multiple logical compression operations and you
    can't guarantee those operations are temporally non-overlapping, you need
    to obtain multiple ``ZstdCompressor`` instances.

    Unless specified otherwise, assume that no two methods of
    ``ZstdCompressor`` instances can be called from multiple Python
    threads simultaneously. In other words, assume instances are not thread safe
    unless stated otherwise.

    :param level:
       Integer compression level. Valid values are all negative integers
       through 22. Lower values generally yield faster operations with lower
       compression ratios. Higher values are generally slower but compress
       better. The default is 3, which is what the ``zstd`` CLI uses. Negative
       levels effectively engage ``--fast`` mode from the ``zstd`` CLI.
    :param dict_data:
       A ``ZstdCompressionDict`` to be used to compress with dictionary
        data.
    :param compression_params:
       A ``ZstdCompressionParameters`` instance defining low-level compression
       parameters. If defined, this will overwrite the ``level`` argument.
    :param write_checksum:
       If True, a 4 byte content checksum will be written with the compressed
       data, allowing the decompressor to perform content verification.
    :param write_content_size:
       If True (the default), the decompressed content size will be included
       in the header of the compressed data. This data will only be written if
       the compressor knows the size of the input data.
    :param write_dict_id:
       Determines whether the dictionary ID will be written into the compressed
       data. Defaults to True. Only adds content to the compressed data if
       a dictionary is being used.
    :param threads:
       Number of threads to use to compress data concurrently. When set,
       compression operations are performed on multiple threads. The default
       value (0) disables multi-threaded compression. A value of ``-1`` means
       to set the number of threads to the number of detected logical CPUs.
    """

    def __init__(self, level=3, dict_data=None, compression_params=None, write_checksum=None, write_content_size=None, write_dict_id=None, threads=0):
        if level > lib.ZSTD_maxCLevel():
            raise ValueError('level must be less than %d' % lib.ZSTD_maxCLevel())
        if threads < 0:
            threads = _cpu_count()
        if compression_params and write_checksum is not None:
            raise ValueError('cannot define compression_params and write_checksum')
        if compression_params and write_content_size is not None:
            raise ValueError('cannot define compression_params and write_content_size')
        if compression_params and write_dict_id is not None:
            raise ValueError('cannot define compression_params and write_dict_id')
        if compression_params and threads:
            raise ValueError('cannot define compression_params and threads')
        if compression_params:
            self._params = _make_cctx_params(compression_params)
        else:
            if write_dict_id is None:
                write_dict_id = True
            params = lib.ZSTD_createCCtxParams()
            if params == ffi.NULL:
                raise MemoryError()
            self._params = ffi.gc(params, lib.ZSTD_freeCCtxParams)
            _set_compression_parameter(self._params, lib.ZSTD_c_compressionLevel, level)
            _set_compression_parameter(self._params, lib.ZSTD_c_contentSizeFlag, write_content_size if write_content_size is not None else 1)
            _set_compression_parameter(self._params, lib.ZSTD_c_checksumFlag, 1 if write_checksum else 0)
            _set_compression_parameter(self._params, lib.ZSTD_c_dictIDFlag, 1 if write_dict_id else 0)
            if threads:
                _set_compression_parameter(self._params, lib.ZSTD_c_nbWorkers, threads)
        cctx = lib.ZSTD_createCCtx()
        if cctx == ffi.NULL:
            raise MemoryError()
        self._cctx = cctx
        self._dict_data = dict_data
        try:
            self._setup_cctx()
        finally:
            self._cctx = ffi.gc(cctx, lib.ZSTD_freeCCtx, size=lib.ZSTD_sizeof_CCtx(cctx))

    def _setup_cctx(self):
        zresult = lib.ZSTD_CCtx_setParametersUsingCCtxParams(self._cctx, self._params)
        if lib.ZSTD_isError(zresult):
            raise ZstdError('could not set compression parameters: %s' % _zstd_error(zresult))
        dict_data = self._dict_data
        if dict_data:
            if dict_data._cdict:
                zresult = lib.ZSTD_CCtx_refCDict(self._cctx, dict_data._cdict)
            else:
                zresult = lib.ZSTD_CCtx_loadDictionary_advanced(self._cctx, dict_data.as_bytes(), len(dict_data), lib.ZSTD_dlm_byRef, dict_data._dict_type)
            if lib.ZSTD_isError(zresult):
                raise ZstdError('could not load compression dictionary: %s' % _zstd_error(zresult))

    def memory_size(self):
        """Obtain the memory usage of this compressor, in bytes.

        >>> cctx = zstandard.ZstdCompressor()
        >>> memory = cctx.memory_size()
        """
        return lib.ZSTD_sizeof_CCtx(self._cctx)

    def compress(self, data):
        """
        Compress data in a single operation.

        This is the simplest mechanism to perform compression: simply pass in a
        value and get a compressed value back. It is almost the most prone to
        abuse.

        The input and output values must fit in memory, so passing in very large
        values can result in excessive memory usage. For this reason, one of the
        streaming based APIs is preferred for larger values.

        :param data:
           Source data to compress
        :return:
           Compressed data

        >>> cctx = zstandard.ZstdCompressor()
        >>> compressed = cctx.compress(b"data to compress")
        """
        lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)
        data_buffer = ffi.from_buffer(data)
        dest_size = lib.ZSTD_compressBound(len(data_buffer))
        out = new_nonzero('char[]', dest_size)
        zresult = lib.ZSTD_CCtx_setPledgedSrcSize(self._cctx, len(data_buffer))
        if lib.ZSTD_isError(zresult):
            raise ZstdError('error setting source size: %s' % _zstd_error(zresult))
        out_buffer = ffi.new('ZSTD_outBuffer *')
        in_buffer = ffi.new('ZSTD_inBuffer *')
        out_buffer.dst = out
        out_buffer.size = dest_size
        out_buffer.pos = 0
        in_buffer.src = data_buffer
        in_buffer.size = len(data_buffer)
        in_buffer.pos = 0
        zresult = lib.ZSTD_compressStream2(self._cctx, out_buffer, in_buffer, lib.ZSTD_e_end)
        if lib.ZSTD_isError(zresult):
            raise ZstdError('cannot compress: %s' % _zstd_error(zresult))
        elif zresult:
            raise ZstdError('unexpected partial frame flush')
        return ffi.buffer(out, out_buffer.pos)[:]

    def compressobj(self, size=-1):
        """
        Obtain a compressor exposing the Python standard library compression API.

        See :py:class:`ZstdCompressionObj` for the full documentation.

        :param size:
           Size in bytes of data that will be compressed.
        :return:
           :py:class:`ZstdCompressionObj`
        """
        lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)
        if size < 0:
            size = lib.ZSTD_CONTENTSIZE_UNKNOWN
        zresult = lib.ZSTD_CCtx_setPledgedSrcSize(self._cctx, size)
        if lib.ZSTD_isError(zresult):
            raise ZstdError('error setting source size: %s' % _zstd_error(zresult))
        cobj = ZstdCompressionObj()
        cobj._out = ffi.new('ZSTD_outBuffer *')
        cobj._dst_buffer = ffi.new('char[]', COMPRESSION_RECOMMENDED_OUTPUT_SIZE)
        cobj._out.dst = cobj._dst_buffer
        cobj._out.size = COMPRESSION_RECOMMENDED_OUTPUT_SIZE
        cobj._out.pos = 0
        cobj._compressor = self
        cobj._finished = False
        return cobj

    def chunker(self, size=-1, chunk_size=COMPRESSION_RECOMMENDED_OUTPUT_SIZE):
        """
        Create an object for iterative compressing to same-sized chunks.

        This API is similar to :py:meth:`ZstdCompressor.compressobj` but has
        better performance properties.

        :param size:
           Size in bytes of data that will be compressed.
        :param chunk_size:
           Size of compressed chunks.
        :return:
           :py:class:`ZstdCompressionChunker`
        """
        lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)
        if size < 0:
            size = lib.ZSTD_CONTENTSIZE_UNKNOWN
        zresult = lib.ZSTD_CCtx_setPledgedSrcSize(self._cctx, size)
        if lib.ZSTD_isError(zresult):
            raise ZstdError('error setting source size: %s' % _zstd_error(zresult))
        return ZstdCompressionChunker(self, chunk_size=chunk_size)

    def copy_stream(self, ifh, ofh, size=-1, read_size=COMPRESSION_RECOMMENDED_INPUT_SIZE, write_size=COMPRESSION_RECOMMENDED_OUTPUT_SIZE):
        """
        Copy data between 2 streams while compressing it.

        Data will be read from ``ifh``, compressed, and written to ``ofh``.
        ``ifh`` must have a ``read(size)`` method. ``ofh`` must have a
        ``write(data)``
        method.

        >>> cctx = zstandard.ZstdCompressor()
        >>> with open(input_path, "rb") as ifh, open(output_path, "wb") as ofh:
        ...     cctx.copy_stream(ifh, ofh)

        It is also possible to declare the size of the source stream:

        >>> cctx = zstandard.ZstdCompressor()
        >>> cctx.copy_stream(ifh, ofh, size=len_of_input)

        You can also specify how large the chunks that are ``read()``
        and ``write()`` from and to the streams:

        >>> cctx = zstandard.ZstdCompressor()
        >>> cctx.copy_stream(ifh, ofh, read_size=32768, write_size=16384)

        The stream copier returns a 2-tuple of bytes read and written:

        >>> cctx = zstandard.ZstdCompressor()
        >>> read_count, write_count = cctx.copy_stream(ifh, ofh)

        :param ifh:
           Source stream to read from
        :param ofh:
           Destination stream to write to
        :param size:
           Size in bytes of the source stream. If defined, compression
           parameters will be tuned for this size.
        :param read_size:
           Chunk sizes that source stream should be ``read()`` from.
        :param write_size:
           Chunk sizes that destination stream should be ``write()`` to.
        :return:
           2-tuple of ints of bytes read and written, respectively.
        """
        if not hasattr(ifh, 'read'):
            raise ValueError('first argument must have a read() method')
        if not hasattr(ofh, 'write'):
            raise ValueError('second argument must have a write() method')
        lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)
        if size < 0:
            size = lib.ZSTD_CONTENTSIZE_UNKNOWN
        zresult = lib.ZSTD_CCtx_setPledgedSrcSize(self._cctx, size)
        if lib.ZSTD_isError(zresult):
            raise ZstdError('error setting source size: %s' % _zstd_error(zresult))
        in_buffer = ffi.new('ZSTD_inBuffer *')
        out_buffer = ffi.new('ZSTD_outBuffer *')
        dst_buffer = ffi.new('char[]', write_size)
        out_buffer.dst = dst_buffer
        out_buffer.size = write_size
        out_buffer.pos = 0
        total_read, total_write = (0, 0)
        while True:
            data = ifh.read(read_size)
            if not data:
                break
            data_buffer = ffi.from_buffer(data)
            total_read += len(data_buffer)
            in_buffer.src = data_buffer
            in_buffer.size = len(data_buffer)
            in_buffer.pos = 0
            while in_buffer.pos < in_buffer.size:
                zresult = lib.ZSTD_compressStream2(self._cctx, out_buffer, in_buffer, lib.ZSTD_e_continue)
                if lib.ZSTD_isError(zresult):
                    raise ZstdError('zstd compress error: %s' % _zstd_error(zresult))
                if out_buffer.pos:
                    ofh.write(ffi.buffer(out_buffer.dst, out_buffer.pos))
                    total_write += out_buffer.pos
                    out_buffer.pos = 0
        while True:
            zresult = lib.ZSTD_compressStream2(self._cctx, out_buffer, in_buffer, lib.ZSTD_e_end)
            if lib.ZSTD_isError(zresult):
                raise ZstdError('error ending compression stream: %s' % _zstd_error(zresult))
            if out_buffer.pos:
                ofh.write(ffi.buffer(out_buffer.dst, out_buffer.pos))
                total_write += out_buffer.pos
                out_buffer.pos = 0
            if zresult == 0:
                break
        return (total_read, total_write)

    def stream_reader(self, source, size=-1, read_size=COMPRESSION_RECOMMENDED_INPUT_SIZE, closefd=True):
        """
        Wrap a readable source with a stream that can read compressed data.

        This will produce an object conforming to the ``io.RawIOBase``
        interface which can be ``read()`` from to retrieve compressed data
        from a source.

        The source object can be any object with a ``read(size)`` method
        or an object that conforms to the buffer protocol.

        See :py:class:`ZstdCompressionReader` for type documentation and usage
        examples.

        :param source:
           Object to read source data from
        :param size:
           Size in bytes of source object.
        :param read_size:
           How many bytes to request when ``read()``'ing from the source.
        :param closefd:
           Whether to close the source stream when the returned stream is
           closed.
        :return:
           :py:class:`ZstdCompressionReader`
        """
        lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)
        try:
            size = len(source)
        except Exception:
            pass
        if size < 0:
            size = lib.ZSTD_CONTENTSIZE_UNKNOWN
        zresult = lib.ZSTD_CCtx_setPledgedSrcSize(self._cctx, size)
        if lib.ZSTD_isError(zresult):
            raise ZstdError('error setting source size: %s' % _zstd_error(zresult))
        return ZstdCompressionReader(self, source, read_size, closefd=closefd)

    def stream_writer(self, writer, size=-1, write_size=COMPRESSION_RECOMMENDED_OUTPUT_SIZE, write_return_read=True, closefd=True):
        """
        Create a stream that will write compressed data into another stream.

        The argument to ``stream_writer()`` must have a ``write(data)`` method.
        As compressed data is available, ``write()`` will be called with the
        compressed data as its argument. Many common Python types implement
        ``write()``, including open file handles and ``io.BytesIO``.

        See :py:class:`ZstdCompressionWriter` for more documentation, including
        usage examples.

        :param writer:
           Stream to write compressed data to.
        :param size:
           Size in bytes of data to be compressed. If set, it will be used
           to influence compression parameter tuning and could result in the
           size being written into the header of the compressed data.
        :param write_size:
           How much data to ``write()`` to ``writer`` at a time.
        :param write_return_read:
           Whether ``write()`` should return the number of bytes that were
           consumed from the input.
        :param closefd:
           Whether to ``close`` the ``writer`` when this stream is closed.
        :return:
           :py:class:`ZstdCompressionWriter`
        """
        if not hasattr(writer, 'write'):
            raise ValueError('must pass an object with a write() method')
        lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)
        if size < 0:
            size = lib.ZSTD_CONTENTSIZE_UNKNOWN
        return ZstdCompressionWriter(self, writer, size, write_size, write_return_read, closefd=closefd)

    def read_to_iter(self, reader, size=-1, read_size=COMPRESSION_RECOMMENDED_INPUT_SIZE, write_size=COMPRESSION_RECOMMENDED_OUTPUT_SIZE):
        """
        Read uncompressed data from a reader and return an iterator

        Returns an iterator of compressed data produced from reading from
        ``reader``.

        This method provides a mechanism to stream compressed data out of a
        source as an iterator of data chunks.

        Uncompressed data will be obtained from ``reader`` by calling the
        ``read(size)`` method of it or by reading a slice (if ``reader``
        conforms to the *buffer protocol*). The source data will be streamed
        into a compressor. As compressed data is available, it will be exposed
        to the iterator.

        Data is read from the source in chunks of ``read_size``. Compressed
        chunks are at most ``write_size`` bytes. Both values default to the
        zstd input and and output defaults, respectively.

        If reading from the source via ``read()``, ``read()`` will be called
        until it raises or returns an empty bytes (``b""``). It is perfectly
        valid for the source to deliver fewer bytes than were what requested
        by ``read(size)``.

        The caller is partially in control of how fast data is fed into the
        compressor by how it consumes the returned iterator. The compressor
        will not consume from the reader unless the caller consumes from the
        iterator.

        >>> cctx = zstandard.ZstdCompressor()
        >>> for chunk in cctx.read_to_iter(fh):
        ...     # Do something with emitted data.

        ``read_to_iter()`` accepts a ``size`` argument declaring the size of
        the input stream:

        >>> cctx = zstandard.ZstdCompressor()
        >>> for chunk in cctx.read_to_iter(fh, size=some_int):
        >>>     pass

        You can also control the size that data is ``read()`` from the source
        and the ideal size of output chunks:

        >>> cctx = zstandard.ZstdCompressor()
        >>> for chunk in cctx.read_to_iter(fh, read_size=16384, write_size=8192):
        >>>     pass

        ``read_to_iter()`` does not give direct control over the sizes of chunks
        fed into the compressor. Instead, chunk sizes will be whatever the object
        being read from delivers. These will often be of a uniform size.

        :param reader:
           Stream providing data to be compressed.
        :param size:
           Size in bytes of input data.
        :param read_size:
           Controls how many bytes are ``read()`` from the source.
        :param write_size:
           Controls the output size of emitted chunks.
        :return:
           Iterator of ``bytes``.
        """
        if hasattr(reader, 'read'):
            have_read = True
        elif hasattr(reader, '__getitem__'):
            have_read = False
            buffer_offset = 0
            size = len(reader)
        else:
            raise ValueError('must pass an object with a read() method or conforms to buffer protocol')
        lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)
        if size < 0:
            size = lib.ZSTD_CONTENTSIZE_UNKNOWN
        zresult = lib.ZSTD_CCtx_setPledgedSrcSize(self._cctx, size)
        if lib.ZSTD_isError(zresult):
            raise ZstdError('error setting source size: %s' % _zstd_error(zresult))
        in_buffer = ffi.new('ZSTD_inBuffer *')
        out_buffer = ffi.new('ZSTD_outBuffer *')
        in_buffer.src = ffi.NULL
        in_buffer.size = 0
        in_buffer.pos = 0
        dst_buffer = ffi.new('char[]', write_size)
        out_buffer.dst = dst_buffer
        out_buffer.size = write_size
        out_buffer.pos = 0
        while True:
            assert out_buffer.pos == 0
            if have_read:
                read_result = reader.read(read_size)
            else:
                remaining = len(reader) - buffer_offset
                slice_size = min(remaining, read_size)
                read_result = reader[buffer_offset:buffer_offset + slice_size]
                buffer_offset += slice_size
            if not read_result:
                break
            read_buffer = ffi.from_buffer(read_result)
            in_buffer.src = read_buffer
            in_buffer.size = len(read_buffer)
            in_buffer.pos = 0
            while in_buffer.pos < in_buffer.size:
                zresult = lib.ZSTD_compressStream2(self._cctx, out_buffer, in_buffer, lib.ZSTD_e_continue)
                if lib.ZSTD_isError(zresult):
                    raise ZstdError('zstd compress error: %s' % _zstd_error(zresult))
                if out_buffer.pos:
                    data = ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
                    out_buffer.pos = 0
                    yield data
            assert out_buffer.pos == 0
            continue
        while True:
            assert out_buffer.pos == 0
            zresult = lib.ZSTD_compressStream2(self._cctx, out_buffer, in_buffer, lib.ZSTD_e_end)
            if lib.ZSTD_isError(zresult):
                raise ZstdError('error ending compression stream: %s' % _zstd_error(zresult))
            if out_buffer.pos:
                data = ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
                out_buffer.pos = 0
                yield data
            if zresult == 0:
                break

    def multi_compress_to_buffer(self, data, threads=-1):
        """
        Compress multiple pieces of data as a single function call.

        (Experimental. Not yet supported by CFFI backend.)

        This function is optimized to perform multiple compression operations
        as as possible with as little overhead as possible.

        Data to be compressed can be passed as a ``BufferWithSegmentsCollection``,
        a ``BufferWithSegments``, or a list containing byte like objects. Each
        element of the container will be compressed individually using the
        configured parameters on the ``ZstdCompressor`` instance.

        The ``threads`` argument controls how many threads to use for
        compression. The default is ``0`` which means to use a single thread.
        Negative values use the number of logical CPUs in the machine.

        The function returns a ``BufferWithSegmentsCollection``. This type
        represents N discrete memory allocations, each holding 1 or more
        compressed frames.

        Output data is written to shared memory buffers. This means that unlike
        regular Python objects, a reference to *any* object within the collection
        keeps the shared buffer and therefore memory backing it alive. This can
        have undesirable effects on process memory usage.

        The API and behavior of this function is experimental and will likely
        change. Known deficiencies include:

        * If asked to use multiple threads, it will always spawn that many
          threads, even if the input is too small to use them. It should
          automatically lower the thread count when the extra threads would
          just add overhead.
        * The buffer allocation strategy is fixed. There is room to make it
          dynamic, perhaps even to allow one output buffer per input,
          facilitating a variation of the API to return a list without the
          adverse effects of shared memory buffers.

        :param data:
           Source to read discrete pieces of data to compress.

           Can be a ``BufferWithSegmentsCollection``, a ``BufferWithSegments``,
           or a ``list[bytes]``.
        :return:
           BufferWithSegmentsCollection holding compressed data.
        """
        raise NotImplementedError()

    def frame_progression(self):
        """
        Return information on how much work the compressor has done.

        Returns a 3-tuple of (ingested, consumed, produced).

        >>> cctx = zstandard.ZstdCompressor()
        >>> (ingested, consumed, produced) = cctx.frame_progression()
        """
        progression = lib.ZSTD_getFrameProgression(self._cctx)
        return (progression.ingested, progression.consumed, progression.produced)