import lz4
import io
import os
import builtins
import sys
from ._frame import (  # noqa: F401
class LZ4FrameDecompressor(object):
    """Create a LZ4 frame decompressor object.

    This can be used to decompress data incrementally.

    For a more convenient way of decompressing an entire compressed frame at
    once, see `lz4.frame.decompress()`.

    Args:
        return_bytearray (bool): When ``False`` a bytes object is returned from
            the calls to methods of this class. When ``True`` a bytearray
            object will be returned. The default is ``False``.

    Attributes:
        eof (bool): ``True`` if the end-of-stream marker has been reached.
            ``False`` otherwise.
        unused_data (bytes): Data found after the end of the compressed stream.
            Before the end of the frame is reached, this will be ``b''``.
        needs_input (bool): ``False`` if the ``decompress()`` method can
            provide more decompressed data before requiring new uncompressed
            input. ``True`` otherwise.

    """

    def __init__(self, return_bytearray=False):
        self._context = create_decompression_context()
        self.eof = False
        self.needs_input = True
        self.unused_data = None
        self._unconsumed_data = b''
        self._return_bytearray = return_bytearray

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self._context = None
        self.eof = None
        self.needs_input = None
        self.unused_data = None
        self._unconsumed_data = None
        self._return_bytearray = None

    def reset(self):
        """Reset the decompressor state.

        This is useful after an error occurs, allowing re-use of the instance.

        """
        reset_decompression_context(self._context)
        self.eof = False
        self.needs_input = True
        self.unused_data = None
        self._unconsumed_data = b''

    def decompress(self, data, max_length=-1):
        """Decompresses part or all of an LZ4 frame of compressed data.

        The returned data should be concatenated with the output of any
        previous calls to `decompress()`.

        If ``max_length`` is non-negative, returns at most ``max_length`` bytes
        of decompressed data. If this limit is reached and further output can
        be produced, the `needs_input` attribute will be set to ``False``. In
        this case, the next call to `decompress()` may provide data as
        ``b''`` to obtain more of the output. In all cases, any unconsumed data
        from previous calls will be prepended to the input data.

        If all of the input ``data`` was decompressed and returned (either
        because this was less than ``max_length`` bytes, or because
        ``max_length`` was negative), the `needs_input` attribute will be set
        to ``True``.

        If an end of frame marker is encountered in the data during
        decompression, decompression will stop at the end of the frame, and any
        data after the end of frame is available from the `unused_data`
        attribute. In this case, the `LZ4FrameDecompressor` instance is reset
        and can be used for further decompression.

        Args:
            data (str, bytes or buffer-compatible object): compressed data to
                decompress

        Keyword Args:
            max_length (int): If this is non-negative, this method returns at
                most ``max_length`` bytes of decompressed data.

        Returns:
            bytes: Uncompressed data

        """
        if not isinstance(data, (bytes, bytearray)):
            data = memoryview(data).tobytes()
        if self._unconsumed_data:
            data = self._unconsumed_data + data
        decompressed, bytes_read, eoframe = decompress_chunk(self._context, data, max_length=max_length, return_bytearray=self._return_bytearray)
        if bytes_read < len(data):
            if eoframe:
                self.unused_data = data[bytes_read:]
            else:
                self._unconsumed_data = data[bytes_read:]
                self.needs_input = False
        else:
            self._unconsumed_data = b''
            self.needs_input = True
            self.unused_data = None
        self.eof = eoframe
        return decompressed