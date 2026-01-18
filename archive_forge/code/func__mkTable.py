import struct
import sys
def _mkTable(poly, n):
    mask = (long(1) << n) - 1
    poly = long(poly) & mask
    table = [_bytecrc(long(i) << n - 8, poly, n) for i in xrange(256)]
    return table