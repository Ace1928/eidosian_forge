import struct
import sys
def _mkTable_r(poly, n):
    mask = (long(1) << n) - 1
    poly = _bitrev(long(poly) & mask, n)
    table = [_bytecrc_r(long(i), poly, n) for i in xrange(256)]
    return table