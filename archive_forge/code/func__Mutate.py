import array
import contextlib
import enum
import struct
def _Mutate(fmt, buf, value, byte_width, value_bit_width):
    if 1 << value_bit_width <= byte_width:
        buf[:byte_width] = _Pack(fmt, value, byte_width)
        return True
    return False