import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def SkipField(buffer, pos, end, tag_bytes):
    """Skips a field with the specified tag.

    |pos| should point to the byte immediately after the tag.

    Returns:
        The new position (after the tag value), or -1 if the tag is an end-group
        tag (in which case the calling loop should break).
    """
    wire_type = ord(tag_bytes[0:1]) & wiretype_mask
    return WIRETYPE_TO_SKIPPER[wire_type](buffer, pos, end)