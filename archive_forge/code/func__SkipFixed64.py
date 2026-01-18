import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def _SkipFixed64(buffer, pos, end):
    """Skip a fixed64 value.  Returns the new position."""
    pos += 8
    if pos > end:
        raise _DecodeError('Truncated message.')
    return pos