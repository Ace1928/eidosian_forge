import struct
import zlib
from .static_tuple import StaticTuple
def _crc32(bit):
    return zlib.crc32(bit) & 4294967295