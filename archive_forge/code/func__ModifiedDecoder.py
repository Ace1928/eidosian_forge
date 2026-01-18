import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def _ModifiedDecoder(wire_type, decode_value, modify_value):
    """Like SimpleDecoder but additionally invokes modify_value on every value
  before storing it.  Usually modify_value is ZigZagDecode.
  """

    def InnerDecode(buffer, pos):
        result, new_pos = decode_value(buffer, pos)
        return (modify_value(result), new_pos)
    return _SimpleDecoder(wire_type, InnerDecode)