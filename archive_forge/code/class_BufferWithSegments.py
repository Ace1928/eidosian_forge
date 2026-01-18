from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
class BufferWithSegments:
    """A memory buffer containing N discrete items of known lengths.

    This type is essentially a fixed size memory address and an array
    of 2-tuples of ``(offset, length)`` 64-bit unsigned native-endian
    integers defining the byte offset and length of each segment within
    the buffer.

    Instances behave like containers.

    Instances also conform to the buffer protocol. So a reference to the
    backing bytes can be obtained via ``memoryview(o)``. A *copy* of the
    backing bytes can be obtained via ``.tobytes()``.

    This type exists to facilitate operations against N>1 items without
    the overhead of Python object creation and management. Used with
    APIs like :py:meth:`ZstdDecompressor.multi_decompress_to_buffer`, it
    is possible to decompress many objects in parallel without the GIL
    held, leading to even better performance.
    """

    @property
    def size(self):
        """Total sizein bytes of the backing buffer."""
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, i):
        """Obtains a segment within the buffer.

        The returned object references memory within this buffer.

        :param i:
           Integer index of segment to retrieve.
        :return:
           :py:class:`BufferSegment`
        """
        raise NotImplementedError()

    def segments(self):
        """Obtain the array of ``(offset, length)`` segments in the buffer.

        :return:
           :py:class:`BufferSegments`
        """
        raise NotImplementedError()

    def tobytes(self):
        """Obtain bytes copy of this instance."""
        raise NotImplementedError()