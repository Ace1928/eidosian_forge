import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def _VarUInt64ByteSizeNoTag(uint64):
    """Returns the number of bytes required to serialize a single varint
  using boundary value comparisons. (unrolled loop optimization -WPierce)
  uint64 must be unsigned.
  """
    if uint64 <= 127:
        return 1
    if uint64 <= 16383:
        return 2
    if uint64 <= 2097151:
        return 3
    if uint64 <= 268435455:
        return 4
    if uint64 <= 34359738367:
        return 5
    if uint64 <= 4398046511103:
        return 6
    if uint64 <= 562949953421311:
        return 7
    if uint64 <= 72057594037927935:
        return 8
    if uint64 <= 9223372036854775807:
        return 9
    if uint64 > UINT64_MAX:
        raise message.EncodeError('Value out of range: %d' % uint64)
    return 10