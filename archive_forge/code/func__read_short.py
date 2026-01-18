import struct
import builtins
import warnings
from collections import namedtuple
def _read_short(file):
    try:
        return struct.unpack('>h', file.read(2))[0]
    except struct.error:
        raise EOFError from None