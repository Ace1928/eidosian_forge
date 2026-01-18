from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def _encode_bytes_little(self, next_value, chunks, tail):
    """helper used by encode_bytes() to handle little-endian encoding"""
    idx = 0
    while idx < chunks:
        v1 = next_value()
        v2 = next_value()
        v3 = next_value()
        yield (v1 & 63)
        yield ((v2 & 15) << 2 | v1 >> 6)
        yield ((v3 & 3) << 4 | v2 >> 4)
        yield (v3 >> 2)
        idx += 1
    if tail:
        v1 = next_value()
        if tail == 1:
            yield (v1 & 63)
            yield (v1 >> 6)
        else:
            assert tail == 2
            v2 = next_value()
            yield (v1 & 63)
            yield ((v2 & 15) << 2 | v1 >> 6)
            yield (v2 >> 4)