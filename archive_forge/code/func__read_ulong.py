import struct
import builtins
import warnings
from collections import namedtuple
def _read_ulong(file):
    try:
        return struct.unpack('>L', file.read(4))[0]
    except struct.error:
        raise EOFError from None