import struct
import sys
def _verifyPoly(poly):
    msg = 'The degree of the polynomial must be 8, 16, 24, 32 or 64'
    poly = long(poly)
    for n in (8, 16, 24, 32, 64):
        low = long(1) << n
        high = low * 2
        if low <= poly < high:
            return n
    raise ValueError(msg)