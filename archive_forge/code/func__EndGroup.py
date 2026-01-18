import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def _EndGroup(buffer, pos, end):
    """Skipping an END_GROUP tag returns -1 to tell the parent loop to break."""
    return -1