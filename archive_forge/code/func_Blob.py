import array
import contextlib
import enum
import struct
@InMap
def Blob(self, value):
    """Encodes binary blob value.

    Args:
      value: A byte/bytearray value to encode

    Returns:
      Offset of the encoded value in underlying the byte buffer.
    """
    return self._WriteBlob(value, append_zero=False, type_=Type.BLOB)