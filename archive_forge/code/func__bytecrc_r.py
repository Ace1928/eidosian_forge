import struct
import sys
def _bytecrc_r(crc, poly, n):
    crc = long(crc)
    poly = long(poly)
    for i in xrange(8):
        if crc & long(1):
            crc = crc >> 1 ^ poly
        else:
            crc = crc >> 1
    mask = (long(1) << n) - 1
    crc = crc & mask
    if mask <= sys.maxint:
        return int(crc)
    return crc