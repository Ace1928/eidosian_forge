import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def _FloatDecoder():
    """Returns a decoder for a float field.

  This code works around a bug in struct.unpack for non-finite 32-bit
  floating-point values.
  """
    local_unpack = struct.unpack

    def InnerDecode(buffer, pos):
        """Decode serialized float to a float and new position.

    Args:
      buffer: memoryview of the serialized bytes
      pos: int, position in the memory view to start at.

    Returns:
      Tuple[float, int] of the deserialized float value and new position
      in the serialized data.
    """
        new_pos = pos + 4
        float_bytes = buffer[pos:new_pos].tobytes()
        if float_bytes[3:4] in b'\x7f\xff' and float_bytes[2:3] >= b'\x80':
            if float_bytes[0:3] != b'\x00\x00\x80':
                return (math.nan, new_pos)
            if float_bytes[3:4] == b'\xff':
                return (-math.inf, new_pos)
            return (math.inf, new_pos)
        result = local_unpack('<f', float_bytes)[0]
        return (result, new_pos)
    return _SimpleDecoder(wire_format.WIRETYPE_FIXED32, InnerDecode)