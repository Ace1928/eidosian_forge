import array
import contextlib
import enum
import struct
@InMap
def IndirectInt(self, value, byte_width=0):
    """Encodes signed integer value indirectly.

    Args:
      value: A signed integer value.
      byte_width: Number of bytes to use: 1, 2, 4, or 8.
    """
    bit_width = BitWidth.I(value) if byte_width == 0 else BitWidth.B(byte_width)
    self._PushIndirect(value, Type.INDIRECT_INT, bit_width)