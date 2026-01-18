import struct
import builtins
import warnings
from collections import namedtuple
def _write_string(f, s):
    if len(s) > 255:
        raise ValueError('string exceeds maximum pstring length')
    f.write(struct.pack('B', len(s)))
    f.write(s)
    if len(s) & 1 == 0:
        f.write(b'\x00')