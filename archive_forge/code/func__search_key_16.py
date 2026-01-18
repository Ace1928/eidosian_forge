import struct
import zlib
from .static_tuple import StaticTuple
def _search_key_16(key):
    """Map the key tuple into a search key string which has 16-way fan out."""
    return b'\x00'.join([b'%08X' % _crc32(bit) for bit in key])