import struct
import builtins
import warnings
from collections import namedtuple
def _write_short(f, x):
    f.write(struct.pack('>h', x))