import struct
import sys
def _verifyParams(poly, initCrc, xorOut):
    sizeBits = _verifyPoly(poly)
    mask = (long(1) << sizeBits) - 1
    initCrc = long(initCrc) & mask
    if mask <= sys.maxint:
        initCrc = int(initCrc)
    xorOut = long(xorOut) & mask
    if mask <= sys.maxint:
        xorOut = int(xorOut)
    return (sizeBits, initCrc, xorOut)