import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def ZigZagDecode(value):
    """Inverse of ZigZagEncode()."""
    if not value & 1:
        return value >> 1
    return value >> 1 ^ ~0