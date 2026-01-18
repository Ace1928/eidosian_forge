import array
import contextlib
import enum
import struct
def _PaddingBytes(buf_size, scalar_size):
    return -buf_size & scalar_size - 1