import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def _DoubleDecoder():
    """Returns a decoder for a double field.

  This code works around a bug in struct.unpack for not-a-number.
  """
    local_unpack = struct.unpack

    def InnerDecode(buffer, pos):
        """Decode serialized double to a double and new position.

    Args:
      buffer: memoryview of the serialized bytes.
      pos: int, position in the memory view to start at.

    Returns:
      Tuple[float, int] of the decoded double value and new position
      in the serialized data.
    """
        new_pos = pos + 8
        double_bytes = buffer[pos:new_pos].tobytes()
        if double_bytes[7:8] in b'\x7f\xff' and double_bytes[6:7] >= b'\xf0' and (double_bytes[0:7] != b'\x00\x00\x00\x00\x00\x00\xf0'):
            return (math.nan, new_pos)
        result = local_unpack('<d', double_bytes)[0]
        return (result, new_pos)
    return _SimpleDecoder(wire_format.WIRETYPE_FIXED64, InnerDecode)