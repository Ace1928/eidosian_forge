import struct
import sys
def _bitrev(x, n):
    x = long(x)
    y = long(0)
    for i in xrange(n):
        y = y << 1 | x & long(1)
        x = x >> 1
    if (long(1) << n) - 1 <= sys.maxint:
        return int(y)
    return y