import struct
from cloudsdk.google.protobuf.internal import wire_format
def _VarintSize(value):
    """Compute the size of a varint value."""
    if value <= 127:
        return 1
    if value <= 16383:
        return 2
    if value <= 2097151:
        return 3
    if value <= 268435455:
        return 4
    if value <= 34359738367:
        return 5
    if value <= 4398046511103:
        return 6
    if value <= 562949953421311:
        return 7
    if value <= 72057594037927935:
        return 8
    if value <= 9223372036854775807:
        return 9
    return 10