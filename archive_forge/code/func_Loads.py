import array
import contextlib
import enum
import struct
def Loads(buf):
    """Returns python object decoded from the buffer."""
    return GetRoot(buf).Value