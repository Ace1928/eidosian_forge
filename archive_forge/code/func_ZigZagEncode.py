import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def ZigZagEncode(value):
    """ZigZag Transform:  Encodes signed integers so that they can be
  effectively used with varint encoding.  See wire_format.h for
  more details.
  """
    if value >= 0:
        return value << 1
    return value << 1 ^ ~0