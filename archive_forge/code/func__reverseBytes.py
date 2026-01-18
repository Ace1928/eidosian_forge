from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
@_memoize
def _reverseBytes(data):
    """
    >>> bin(ord(_reverseBytes(0b00100111)))
    '0b11100100'
    >>> _reverseBytes(b'\\x00\\xf0')
    b'\\x00\\x0f'
    """
    if isinstance(data, bytes) and len(data) != 1:
        return bytesjoin(map(_reverseBytes, data))
    byte = byteord(data)
    result = 0
    for i in range(8):
        result = result << 1
        result |= byte & 1
        byte = byte >> 1
    return bytechr(result)