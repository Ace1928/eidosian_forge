from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def _decode_bytes_little(self, next_value, chunks, tail):
    """helper used by decode_bytes() to handle little-endian encoding"""
    idx = 0
    while idx < chunks:
        v1 = next_value()
        v2 = next_value()
        v3 = next_value()
        v4 = next_value()
        yield (v1 | (v2 & 3) << 6)
        yield (v2 >> 2 | (v3 & 15) << 4)
        yield (v3 >> 4 | v4 << 2)
        idx += 1
    if tail:
        v1 = next_value()
        v2 = next_value()
        yield (v1 | (v2 & 3) << 6)
        if tail == 3:
            v3 = next_value()
            yield (v2 >> 2 | (v3 & 15) << 4)