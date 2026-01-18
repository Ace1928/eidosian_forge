import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def _StructPackDecoder(wire_type, format):
    """Return a constructor for a decoder for a fixed-width field.

  Args:
      wire_type:  The field's wire type.
      format:  The format string to pass to struct.unpack().
  """
    value_size = struct.calcsize(format)
    local_unpack = struct.unpack

    def InnerDecode(buffer, pos):
        new_pos = pos + value_size
        result = local_unpack(format, buffer[pos:new_pos])[0]
        return (result, new_pos)
    return _SimpleDecoder(wire_type, InnerDecode)