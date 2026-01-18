import collections
import io   # For io.BytesIO
import itertools
import math
import operator
import re
import struct
import sys
import warnings
import zlib
from array import array
fromarray = from_array
def _iter_straight_packed(self, byte_blocks):
    """Iterator that undoes the effect of filtering;
        yields each row as a sequence of packed bytes.
        Assumes input is straightlaced.
        `byte_blocks` should be an iterable that yields the raw bytes
        in blocks of arbitrary size.
        """
    rb = self.row_bytes
    a = bytearray()
    recon = None
    for some_bytes in byte_blocks:
        a.extend(some_bytes)
        while len(a) >= rb + 1:
            filter_type = a[0]
            scanline = a[1:rb + 1]
            del a[:rb + 1]
            recon = self.undo_filter(filter_type, scanline, recon)
            yield recon
    if len(a) != 0:
        raise FormatError('Wrong size for decompressed IDAT chunk.')
    assert len(a) == 0