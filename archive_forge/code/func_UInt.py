import array
import contextlib
import enum
import struct
@InMap
def UInt(self, value, byte_width=0):
    """Encodes unsigned integer value.

    Args:
      value: An unsigned integer value.
      byte_width: Number of bytes to use: 1, 2, 4, or 8.
    """
    bit_width = BitWidth.U(value) if byte_width == 0 else BitWidth.B(byte_width)
    self._stack.append(Value.UInt(value, bit_width))