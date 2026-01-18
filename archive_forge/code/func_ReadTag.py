import math
import struct
from cloudsdk.google.protobuf.internal import containers
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import message
def ReadTag(buffer, pos):
    """Read a tag from the memoryview, and return a (tag_bytes, new_pos) tuple.

  We return the raw bytes of the tag rather than decoding them.  The raw
  bytes can then be used to look up the proper decoder.  This effectively allows
  us to trade some work that would be done in pure-python (decoding a varint)
  for work that is done in C (searching for a byte string in a hash table).
  In a low-level language it would be much cheaper to decode the varint and
  use that, but not in Python.

  Args:
    buffer: memoryview object of the encoded bytes
    pos: int of the current position to start from

  Returns:
    Tuple[bytes, int] of the tag data and new position.
  """
    start = pos
    while buffer[pos] & 128:
        pos += 1
    pos += 1
    tag_bytes = buffer[start:pos].tobytes()
    return (tag_bytes, pos)