from collections import namedtuple
import warnings
def _read_u32(file):
    x = 0
    for i in range(4):
        byte = file.read(1)
        if not byte:
            raise EOFError
        x = x * 256 + ord(byte)
    return x