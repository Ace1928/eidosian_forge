import _imp
import _io
import sys
import _warnings
import marshal
def _unpack_uint16(data):
    """Convert 2 bytes in little-endian to an integer."""
    assert len(data) == 2
    return int.from_bytes(data, 'little')