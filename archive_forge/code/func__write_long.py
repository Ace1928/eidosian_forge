import struct
import builtins
import warnings
from collections import namedtuple
def _write_long(f, x):
    f.write(struct.pack('>l', x))