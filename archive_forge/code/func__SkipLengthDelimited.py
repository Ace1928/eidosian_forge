import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def _SkipLengthDelimited(buffer, pos, end):
    """Skip a length-delimited value.  Returns the new position."""
    size, pos = _DecodeVarint(buffer, pos)
    pos += size
    if pos > end:
        raise _DecodeError('Truncated message.')
    return pos