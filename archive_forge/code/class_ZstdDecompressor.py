from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
class ZstdDecompressor(object):
    """
    Context for performing zstandard decompression.

    Each instance is essentially a wrapper around a ``ZSTD_DCtx`` from zstd's
    C API.

    An instance can compress data various ways. Instances can be used multiple
    times.

    The interface of this class is very similar to
    :py:class:`zstandard.ZstdCompressor` (by design).

    Assume that each ``ZstdDecompressor`` instance can only handle a single
    logical compression operation at the same time. i.e. if you call a method
    like ``decompressobj()`` to obtain multiple objects derived from the same
    ``ZstdDecompressor`` instance and attempt to use them simultaneously, errors
    will likely occur.

    If you need to perform multiple logical decompression operations and you
    can't guarantee those operations are temporally non-overlapping, you need
    to obtain multiple ``ZstdDecompressor`` instances.

    Unless specified otherwise, assume that no two methods of
    ``ZstdDecompressor`` instances can be called from multiple Python
    threads simultaneously. In other words, assume instances are not thread safe
    unless stated otherwise.

    :param dict_data:
       Compression dictionary to use.
    :param max_window_size:
       Sets an upper limit on the window size for decompression operations in
       kibibytes. This setting can be used to prevent large memory allocations
       for inputs using large compression windows.
    :param format:
       Set the format of data for the decoder.

       By default this is ``zstandard.FORMAT_ZSTD1``. It can be set to
       ``zstandard.FORMAT_ZSTD1_MAGICLESS`` to allow decoding frames without
       the 4 byte magic header. Not all decompression APIs support this mode.
    """

    def __init__(self, dict_data=None, max_window_size=0, format=FORMAT_ZSTD1):
        self._dict_data = dict_data
        self._max_window_size = max_window_size
        self._format = format
        dctx = lib.ZSTD_createDCtx()
        if dctx == ffi.NULL:
            raise MemoryError()
        self._dctx = dctx
        try:
            self._ensure_dctx()
        finally:
            self._dctx = ffi.gc(dctx, lib.ZSTD_freeDCtx, size=lib.ZSTD_sizeof_DCtx(dctx))

    def memory_size(self):
        """Size of decompression context, in bytes.

        >>> dctx = zstandard.ZstdDecompressor()
        >>> size = dctx.memory_size()
        """
        return lib.ZSTD_sizeof_DCtx(self._dctx)

    def decompress(self, data, max_output_size=0, read_across_frames=False, allow_extra_data=True):
        """
        Decompress data in a single operation.

        This method will decompress the input data in a single operation and
        return the decompressed data.

        The input bytes are expected to contain at least 1 full Zstandard frame
        (something compressed with :py:meth:`ZstdCompressor.compress` or
        similar). If the input does not contain a full frame, an exception will
        be raised.

        ``read_across_frames`` controls whether to read multiple zstandard
        frames in the input. When False, decompression stops after reading the
        first frame. This feature is not yet implemented but the argument is
        provided for forward API compatibility when the default is changed to
        True in a future release. For now, if you need to decompress multiple
        frames, use an API like :py:meth:`ZstdCompressor.stream_reader` with
        ``read_across_frames=True``.

        ``allow_extra_data`` controls how to handle extra input data after a
        fully decoded frame. If False, any extra data (which could be a valid
        zstd frame) will result in ``ZstdError`` being raised. If True, extra
        data is silently ignored. The default will likely change to False in a
        future release when ``read_across_frames`` defaults to True.

        If the input contains extra data after a full frame, that extra input
        data is silently ignored. This behavior is undesirable in many scenarios
        and will likely be changed or controllable in a future release (see
        #181).

        If the frame header of the compressed data does not contain the content
        size, ``max_output_size`` must be specified or ``ZstdError`` will be
        raised. An allocation of size ``max_output_size`` will be performed and an
        attempt will be made to perform decompression into that buffer. If the
        buffer is too small or cannot be allocated, ``ZstdError`` will be
        raised. The buffer will be resized if it is too large.

        Uncompressed data could be much larger than compressed data. As a result,
        calling this function could result in a very large memory allocation
        being performed to hold the uncompressed data. This could potentially
        result in ``MemoryError`` or system memory swapping. If you don't need
        the full output data in a single contiguous array in memory, consider
        using streaming decompression for more resilient memory behavior.

        Usage:

        >>> dctx = zstandard.ZstdDecompressor()
        >>> decompressed = dctx.decompress(data)

        If the compressed data doesn't have its content size embedded within it,
        decompression can be attempted by specifying the ``max_output_size``
        argument:

        >>> dctx = zstandard.ZstdDecompressor()
        >>> uncompressed = dctx.decompress(data, max_output_size=1048576)

        Ideally, ``max_output_size`` will be identical to the decompressed
        output size.

        .. important::

           If the exact size of decompressed data is unknown (not passed in
           explicitly and not stored in the zstd frame), for performance
           reasons it is encouraged to use a streaming API.

        :param data:
           Compressed data to decompress.
        :param max_output_size:
           Integer max size of response.

           If ``0``, there is no limit and we can attempt to allocate an output
           buffer of infinite size.
        :return:
           ``bytes`` representing decompressed output.
        """
        if read_across_frames:
            raise ZstdError('ZstdDecompressor.read_across_frames=True is not yet implemented')
        self._ensure_dctx()
        data_buffer = ffi.from_buffer(data)
        output_size = lib.ZSTD_getFrameContentSize(data_buffer, len(data_buffer))
        if output_size == lib.ZSTD_CONTENTSIZE_ERROR:
            raise ZstdError('error determining content size from frame header')
        elif output_size == 0:
            return b''
        elif output_size == lib.ZSTD_CONTENTSIZE_UNKNOWN:
            if not max_output_size:
                raise ZstdError('could not determine content size in frame header')
            result_buffer = ffi.new('char[]', max_output_size)
            result_size = max_output_size
            output_size = 0
        else:
            result_buffer = ffi.new('char[]', output_size)
            result_size = output_size
        out_buffer = ffi.new('ZSTD_outBuffer *')
        out_buffer.dst = result_buffer
        out_buffer.size = result_size
        out_buffer.pos = 0
        in_buffer = ffi.new('ZSTD_inBuffer *')
        in_buffer.src = data_buffer
        in_buffer.size = len(data_buffer)
        in_buffer.pos = 0
        zresult = lib.ZSTD_decompressStream(self._dctx, out_buffer, in_buffer)
        if lib.ZSTD_isError(zresult):
            raise ZstdError('decompression error: %s' % _zstd_error(zresult))
        elif zresult:
            raise ZstdError('decompression error: did not decompress full frame')
        elif output_size and out_buffer.pos != output_size:
            raise ZstdError('decompression error: decompressed %d bytes; expected %d' % (zresult, output_size))
        elif not allow_extra_data and in_buffer.pos < in_buffer.size:
            count = in_buffer.size - in_buffer.pos
            raise ZstdError('compressed input contains %d bytes of unused data, which is disallowed' % count)
        return ffi.buffer(result_buffer, out_buffer.pos)[:]

    def stream_reader(self, source, read_size=DECOMPRESSION_RECOMMENDED_INPUT_SIZE, read_across_frames=False, closefd=True):
        """
        Read-only stream wrapper that performs decompression.

        This method obtains an object that conforms to the ``io.RawIOBase``
        interface and performs transparent decompression via ``read()``
        operations. Source data is obtained by calling ``read()`` on a
        source stream or object implementing the buffer protocol.

        See :py:class:`zstandard.ZstdDecompressionReader` for more documentation
        and usage examples.

        :param source:
           Source of compressed data to decompress. Can be any object
           with a ``read(size)`` method or that conforms to the buffer protocol.
        :param read_size:
           Integer number of bytes to read from the source and feed into the
           compressor at a time.
        :param read_across_frames:
           Whether to read data across multiple zstd frames. If False,
           decompression is stopped at frame boundaries.
        :param closefd:
           Whether to close the source stream when this instance is closed.
        :return:
           :py:class:`zstandard.ZstdDecompressionReader`.
        """
        self._ensure_dctx()
        return ZstdDecompressionReader(self, source, read_size, read_across_frames, closefd=closefd)

    def decompressobj(self, write_size=DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE, read_across_frames=False):
        """Obtain a standard library compatible incremental decompressor.

        See :py:class:`ZstdDecompressionObj` for more documentation
        and usage examples.

        :param write_size: size of internal output buffer to collect decompressed
          chunks in.
        :param read_across_frames: whether to read across multiple zstd frames.
          If False, reading stops after 1 frame and subsequent decompress
          attempts will raise an exception.
        :return:
           :py:class:`zstandard.ZstdDecompressionObj`
        """
        if write_size < 1:
            raise ValueError('write_size must be positive')
        self._ensure_dctx()
        return ZstdDecompressionObj(self, write_size=write_size, read_across_frames=read_across_frames)

    def read_to_iter(self, reader, read_size=DECOMPRESSION_RECOMMENDED_INPUT_SIZE, write_size=DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE, skip_bytes=0):
        """Read compressed data to an iterator of uncompressed chunks.

        This method will read data from ``reader``, feed it to a decompressor,
        and emit ``bytes`` chunks representing the decompressed result.

        >>> dctx = zstandard.ZstdDecompressor()
        >>> for chunk in dctx.read_to_iter(fh):
        ...     # Do something with original data.

        ``read_to_iter()`` accepts an object with a ``read(size)`` method that
        will return compressed bytes or an object conforming to the buffer
        protocol.

        ``read_to_iter()`` returns an iterator whose elements are chunks of the
        decompressed data.

        The size of requested ``read()`` from the source can be specified:

        >>> dctx = zstandard.ZstdDecompressor()
        >>> for chunk in dctx.read_to_iter(fh, read_size=16384):
        ...    pass

        It is also possible to skip leading bytes in the input data:

        >>> dctx = zstandard.ZstdDecompressor()
        >>> for chunk in dctx.read_to_iter(fh, skip_bytes=1):
        ...    pass

        .. tip::

           Skipping leading bytes is useful if the source data contains extra
           *header* data. Traditionally, you would need to create a slice or
           ``memoryview`` of the data you want to decompress. This would create
           overhead. It is more efficient to pass the offset into this API.

        Similarly to :py:meth:`ZstdCompressor.read_to_iter`, the consumer of the
        iterator controls when data is decompressed. If the iterator isn't consumed,
        decompression is put on hold.

        When ``read_to_iter()`` is passed an object conforming to the buffer protocol,
        the behavior may seem similar to what occurs when the simple decompression
        API is used. However, this API works when the decompressed size is unknown.
        Furthermore, if feeding large inputs, the decompressor will work in chunks
        instead of performing a single operation.

        :param reader:
           Source of compressed data. Can be any object with a
           ``read(size)`` method or any object conforming to the buffer
           protocol.
        :param read_size:
           Integer size of data chunks to read from ``reader`` and feed into
           the decompressor.
        :param write_size:
           Integer size of data chunks to emit from iterator.
        :param skip_bytes:
           Integer number of bytes to skip over before sending data into
           the decompressor.
        :return:
           Iterator of ``bytes`` representing uncompressed data.
        """
        if skip_bytes >= read_size:
            raise ValueError('skip_bytes must be smaller than read_size')
        if hasattr(reader, 'read'):
            have_read = True
        elif hasattr(reader, '__getitem__'):
            have_read = False
            buffer_offset = 0
            size = len(reader)
        else:
            raise ValueError('must pass an object with a read() method or conforms to buffer protocol')
        if skip_bytes:
            if have_read:
                reader.read(skip_bytes)
            else:
                if skip_bytes > size:
                    raise ValueError('skip_bytes larger than first input chunk')
                buffer_offset = skip_bytes
        self._ensure_dctx()
        in_buffer = ffi.new('ZSTD_inBuffer *')
        out_buffer = ffi.new('ZSTD_outBuffer *')
        dst_buffer = ffi.new('char[]', write_size)
        out_buffer.dst = dst_buffer
        out_buffer.size = len(dst_buffer)
        out_buffer.pos = 0
        while True:
            assert out_buffer.pos == 0
            if have_read:
                read_result = reader.read(read_size)
            else:
                remaining = size - buffer_offset
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
                assert out_buffer.pos == 0
                zresult = lib.ZSTD_decompressStream(self._dctx, out_buffer, in_buffer)
                if lib.ZSTD_isError(zresult):
                    raise ZstdError('zstd decompress error: %s' % _zstd_error(zresult))
                if out_buffer.pos:
                    data = ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
                    out_buffer.pos = 0
                    yield data
                if zresult == 0:
                    return
            continue

    def stream_writer(self, writer, write_size=DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE, write_return_read=True, closefd=True):
        """
        Push-based stream wrapper that performs decompression.

        This method constructs a stream wrapper that conforms to the
        ``io.RawIOBase`` interface and performs transparent decompression
        when writing to a wrapper stream.

        See :py:class:`zstandard.ZstdDecompressionWriter` for more documentation
        and usage examples.

        :param writer:
           Destination for decompressed output. Can be any object with a
           ``write(data)``.
        :param write_size:
           Integer size of chunks to ``write()`` to ``writer``.
        :param write_return_read:
           Whether ``write()`` should return the number of bytes of input
           consumed. If False, ``write()`` returns the number of bytes sent
           to the inner stream.
        :param closefd:
           Whether to ``close()`` the inner stream when this stream is closed.
        :return:
           :py:class:`zstandard.ZstdDecompressionWriter`
        """
        if not hasattr(writer, 'write'):
            raise ValueError('must pass an object with a write() method')
        return ZstdDecompressionWriter(self, writer, write_size, write_return_read, closefd=closefd)

    def copy_stream(self, ifh, ofh, read_size=DECOMPRESSION_RECOMMENDED_INPUT_SIZE, write_size=DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE):
        """
        Copy data between streams, decompressing in the process.

        Compressed data will be read from ``ifh``, decompressed, and written
        to ``ofh``.

        >>> dctx = zstandard.ZstdDecompressor()
        >>> dctx.copy_stream(ifh, ofh)

        e.g. to decompress a file to another file:

        >>> dctx = zstandard.ZstdDecompressor()
        >>> with open(input_path, 'rb') as ifh, open(output_path, 'wb') as ofh:
        ...     dctx.copy_stream(ifh, ofh)

        The size of chunks being ``read()`` and ``write()`` from and to the
        streams can be specified:

        >>> dctx = zstandard.ZstdDecompressor()
        >>> dctx.copy_stream(ifh, ofh, read_size=8192, write_size=16384)

        :param ifh:
           Source stream to read compressed data from.

           Must have a ``read()`` method.
        :param ofh:
           Destination stream to write uncompressed data to.

           Must have a ``write()`` method.
        :param read_size:
           The number of bytes to ``read()`` from the source in a single
           operation.
        :param write_size:
           The number of bytes to ``write()`` to the destination in a single
           operation.
        :return:
           2-tuple of integers representing the number of bytes read and
           written, respectively.
        """
        if not hasattr(ifh, 'read'):
            raise ValueError('first argument must have a read() method')
        if not hasattr(ofh, 'write'):
            raise ValueError('second argument must have a write() method')
        self._ensure_dctx()
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
                zresult = lib.ZSTD_decompressStream(self._dctx, out_buffer, in_buffer)
                if lib.ZSTD_isError(zresult):
                    raise ZstdError('zstd decompressor error: %s' % _zstd_error(zresult))
                if out_buffer.pos:
                    ofh.write(ffi.buffer(out_buffer.dst, out_buffer.pos))
                    total_write += out_buffer.pos
                    out_buffer.pos = 0
        return (total_read, total_write)

    def decompress_content_dict_chain(self, frames):
        """
        Decompress a series of frames using the content dictionary chaining technique.

        Such a list of frames is produced by compressing discrete inputs where
        each non-initial input is compressed with a *prefix* dictionary consisting
        of the content of the previous input.

        For example, say you have the following inputs:

        >>> inputs = [b"input 1", b"input 2", b"input 3"]

        The zstd frame chain consists of:

        1. ``b"input 1"`` compressed in standalone/discrete mode
        2. ``b"input 2"`` compressed using ``b"input 1"`` as a *prefix* dictionary
        3. ``b"input 3"`` compressed using ``b"input 2"`` as a *prefix* dictionary

        Each zstd frame **must** have the content size written.

        The following Python code can be used to produce a *prefix dictionary chain*:

        >>> def make_chain(inputs):
        ...    frames = []
        ...
        ...    # First frame is compressed in standalone/discrete mode.
        ...    zctx = zstandard.ZstdCompressor()
        ...    frames.append(zctx.compress(inputs[0]))
        ...
        ...    # Subsequent frames use the previous fulltext as a prefix dictionary
        ...    for i, raw in enumerate(inputs[1:]):
        ...        dict_data = zstandard.ZstdCompressionDict(
        ...            inputs[i], dict_type=zstandard.DICT_TYPE_RAWCONTENT)
        ...        zctx = zstandard.ZstdCompressor(dict_data=dict_data)
        ...        frames.append(zctx.compress(raw))
        ...
        ...    return frames

        ``decompress_content_dict_chain()`` returns the uncompressed data of the last
        element in the input chain.

        .. note::

           It is possible to implement *prefix dictionary chain* decompression
           on top of other APIs. However, this function will likely be faster -
           especially for long input chains - as it avoids the overhead of
           instantiating and passing around intermediate objects between
           multiple functions.

        :param frames:
           List of ``bytes`` holding compressed zstd frames.
        :return:
        """
        if not isinstance(frames, list):
            raise TypeError('argument must be a list')
        if not frames:
            raise ValueError('empty input chain')
        chunk = frames[0]
        if not isinstance(chunk, bytes):
            raise ValueError('chunk 0 must be bytes')
        chunk_buffer = ffi.from_buffer(chunk)
        params = ffi.new('ZSTD_frameHeader *')
        zresult = lib.ZSTD_getFrameHeader(params, chunk_buffer, len(chunk_buffer))
        if lib.ZSTD_isError(zresult):
            raise ValueError('chunk 0 is not a valid zstd frame')
        elif zresult:
            raise ValueError('chunk 0 is too small to contain a zstd frame')
        if params.frameContentSize == lib.ZSTD_CONTENTSIZE_UNKNOWN:
            raise ValueError('chunk 0 missing content size in frame')
        self._ensure_dctx(load_dict=False)
        last_buffer = ffi.new('char[]', params.frameContentSize)
        out_buffer = ffi.new('ZSTD_outBuffer *')
        out_buffer.dst = last_buffer
        out_buffer.size = len(last_buffer)
        out_buffer.pos = 0
        in_buffer = ffi.new('ZSTD_inBuffer *')
        in_buffer.src = chunk_buffer
        in_buffer.size = len(chunk_buffer)
        in_buffer.pos = 0
        zresult = lib.ZSTD_decompressStream(self._dctx, out_buffer, in_buffer)
        if lib.ZSTD_isError(zresult):
            raise ZstdError('could not decompress chunk 0: %s' % _zstd_error(zresult))
        elif zresult:
            raise ZstdError('chunk 0 did not decompress full frame')
        if len(frames) == 1:
            return ffi.buffer(last_buffer, len(last_buffer))[:]
        i = 1
        while i < len(frames):
            chunk = frames[i]
            if not isinstance(chunk, bytes):
                raise ValueError('chunk %d must be bytes' % i)
            chunk_buffer = ffi.from_buffer(chunk)
            zresult = lib.ZSTD_getFrameHeader(params, chunk_buffer, len(chunk_buffer))
            if lib.ZSTD_isError(zresult):
                raise ValueError('chunk %d is not a valid zstd frame' % i)
            elif zresult:
                raise ValueError('chunk %d is too small to contain a zstd frame' % i)
            if params.frameContentSize == lib.ZSTD_CONTENTSIZE_UNKNOWN:
                raise ValueError('chunk %d missing content size in frame' % i)
            dest_buffer = ffi.new('char[]', params.frameContentSize)
            out_buffer.dst = dest_buffer
            out_buffer.size = len(dest_buffer)
            out_buffer.pos = 0
            in_buffer.src = chunk_buffer
            in_buffer.size = len(chunk_buffer)
            in_buffer.pos = 0
            zresult = lib.ZSTD_decompressStream(self._dctx, out_buffer, in_buffer)
            if lib.ZSTD_isError(zresult):
                raise ZstdError('could not decompress chunk %d: %s' % _zstd_error(zresult))
            elif zresult:
                raise ZstdError('chunk %d did not decompress full frame' % i)
            last_buffer = dest_buffer
            i += 1
        return ffi.buffer(last_buffer, len(last_buffer))[:]

    def multi_decompress_to_buffer(self, frames, decompressed_sizes=None, threads=0):
        """
        Decompress multiple zstd frames to output buffers as a single operation.

        (Experimental. Not available in CFFI backend.)

        Compressed frames can be passed to the function as a
        ``BufferWithSegments``, a ``BufferWithSegmentsCollection``, or as a
        list containing objects that conform to the buffer protocol. For best
        performance, pass a ``BufferWithSegmentsCollection`` or a
        ``BufferWithSegments``, as minimal input validation will be done for
        that type. If calling from Python (as opposed to C), constructing one
        of these instances may add overhead cancelling out the performance
        overhead of validation for list inputs.

        Returns a ``BufferWithSegmentsCollection`` containing the decompressed
        data. All decompressed data is allocated in a single memory buffer. The
        ``BufferWithSegments`` instance tracks which objects are at which offsets
        and their respective lengths.

        >>> dctx = zstandard.ZstdDecompressor()
        >>> results = dctx.multi_decompress_to_buffer([b'...', b'...'])

        The decompressed size of each frame MUST be discoverable. It can either be
        embedded within the zstd frame or passed in via the ``decompressed_sizes``
        argument.

        The ``decompressed_sizes`` argument is an object conforming to the buffer
        protocol which holds an array of 64-bit unsigned integers in the machine's
        native format defining the decompressed sizes of each frame. If this argument
        is passed, it avoids having to scan each frame for its decompressed size.
        This frame scanning can add noticeable overhead in some scenarios.

        >>> frames = [...]
        >>> sizes = struct.pack('=QQQQ', len0, len1, len2, len3)
        >>>
        >>> dctx = zstandard.ZstdDecompressor()
        >>> results = dctx.multi_decompress_to_buffer(frames, decompressed_sizes=sizes)

        .. note::

           It is possible to pass a ``mmap.mmap()`` instance into this function by
           wrapping it with a ``BufferWithSegments`` instance (which will define the
           offsets of frames within the memory mapped region).

        This function is logically equivalent to performing
        :py:meth:`ZstdCompressor.decompress` on each input frame and returning the
        result.

        This function exists to perform decompression on multiple frames as fast
        as possible by having as little overhead as possible. Since decompression is
        performed as a single operation and since the decompressed output is stored in
        a single buffer, extra memory allocations, Python objects, and Python function
        calls are avoided. This is ideal for scenarios where callers know up front that
        they need to access data for multiple frames, such as when  *delta chains* are
        being used.

        Currently, the implementation always spawns multiple threads when requested,
        even if the amount of work to do is small. In the future, it will be smarter
        about avoiding threads and their associated overhead when the amount of
        work to do is small.

        :param frames:
           Source defining zstd frames to decompress.
        :param decompressed_sizes:
           Array of integers representing sizes of decompressed zstd frames.
        :param threads:
           How many threads to use for decompression operations.

           Negative values will use the same number of threads as logical CPUs
           on the machine. Values ``0`` or ``1`` use a single thread.
        :return:
           ``BufferWithSegmentsCollection``
        """
        raise NotImplementedError()

    def _ensure_dctx(self, load_dict=True):
        lib.ZSTD_DCtx_reset(self._dctx, lib.ZSTD_reset_session_only)
        if self._max_window_size:
            zresult = lib.ZSTD_DCtx_setMaxWindowSize(self._dctx, self._max_window_size)
            if lib.ZSTD_isError(zresult):
                raise ZstdError('unable to set max window size: %s' % _zstd_error(zresult))
        zresult = lib.ZSTD_DCtx_setParameter(self._dctx, lib.ZSTD_d_format, self._format)
        if lib.ZSTD_isError(zresult):
            raise ZstdError('unable to set decoding format: %s' % _zstd_error(zresult))
        if self._dict_data and load_dict:
            zresult = lib.ZSTD_DCtx_refDDict(self._dctx, self._dict_data._ddict)
            if lib.ZSTD_isError(zresult):
                raise ZstdError('unable to reference prepared dictionary: %s' % _zstd_error(zresult))