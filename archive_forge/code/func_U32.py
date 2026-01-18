import struct
import zlib
def U32(i):
    """Return i as an unsigned integer, assuming it fits in 32 bits.

    If it's >= 2GB when viewed as a 32-bit unsigned int, return a long.
    """
    if i < 0:
        i += 1 << 32
    return i