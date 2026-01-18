from __future__ import absolute_import, division, print_function
import bz2
import hashlib
import logging
import os
import re
import struct
import sys
import types
import zlib
from io import BytesIO
class ListStream(BaseStream):
    """A streamable list object used for writing or reading.
    In read mode, it can also be iterated over.
    """

    @property
    def count(self):
        """The number of elements in the stream (can be -1 for unclosed
        streams in read-mode).
        """
        return self._count

    @property
    def index(self):
        """The current index of the element to read/write."""
        return self._i

    def append(self, item):
        """Append an item to the streaming list. The object is immediately
        serialized and written to the underlying file.
        """
        if self._count != self._i:
            raise IOError('Can only append items to the end of the stream.')
        if self._f is None:
            raise IOError('List stream is not associated with a file yet.')
        if self._f.closed:
            raise IOError('Cannot stream to a close file.')
        self._encode(self._f, item, [self], None)
        self._i += 1
        self._count += 1

    def close(self, unstream=False):
        """Close the stream, marking the number of written elements. New
        elements may still be appended, but they won't be read during decoding.
        If ``unstream`` is False, the stream is turned into a regular list
        (not streaming).
        """
        if self._count != self._i:
            raise IOError('Can only close when at the end of the stream.')
        if self._f is None:
            raise IOError('ListStream is not associated with a file yet.')
        if self._f.closed:
            raise IOError('Cannot close a stream on a close file.')
        i = self._f.tell()
        self._f.seek(self._start_pos - 8 - 1)
        self._f.write(spack('<B', 253 if unstream else 254))
        self._f.write(spack('<Q', self._count))
        self._f.seek(i)

    def next(self):
        """Read and return the next element in the streaming list.
        Raises StopIteration if the stream is exhausted.
        """
        if self._mode != 'r':
            raise IOError('This ListStream in not in read mode.')
        if self._f is None:
            raise IOError('ListStream is not associated with a file yet.')
        if getattr(self._f, 'closed', None):
            raise IOError('Cannot read a stream from a close file.')
        if self._count >= 0:
            if self._i >= self._count:
                raise StopIteration()
            self._i += 1
            return self._decode(self._f)
        else:
            try:
                res = self._decode(self._f)
                self._i += 1
                return res
            except EOFError:
                self._count = self._i
                raise StopIteration()

    def __iter__(self):
        if self._mode != 'r':
            raise IOError('Cannot iterate: ListStream in not in read mode.')
        return self

    def __next__(self):
        return self.next()