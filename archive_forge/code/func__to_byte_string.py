import base64
import uuid
from heat.common.i18n import _
def _to_byte_string(value, num_bits):
    """Convert an integer to a big-endian string of bytes with padding.

    Padding is added at the end (i.e. after the least-significant bit) if
    required.
    """
    shifts = range(num_bits - 8, -8, -8)

    def byte_at(off):
        return (value >> off if off >= 0 else value << -off) & 255
    return b''.join((bytes((byte_at(offset),)) for offset in shifts))