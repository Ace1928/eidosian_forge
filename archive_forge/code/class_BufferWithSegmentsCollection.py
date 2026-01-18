from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
class BufferWithSegmentsCollection:
    """A virtual spanning view over multiple BufferWithSegments.

    Instances are constructed from 1 or more :py:class:`BufferWithSegments`
    instances. The resulting object behaves like an ordered sequence whose
    members are the segments within each ``BufferWithSegments``.

    If the object is composed of 2 ``BufferWithSegments`` instances with the
    first having 2 segments and the second have 3 segments, then ``b[0]``
    and ``b[1]`` access segments in the first object and ``b[2]``, ``b[3]``,
    and ``b[4]`` access segments from the second.
    """

    def __len__(self):
        """The number of segments within all ``BufferWithSegments``."""
        raise NotImplementedError()

    def __getitem__(self, i):
        """Obtain the ``BufferSegment`` at an offset."""
        raise NotImplementedError()