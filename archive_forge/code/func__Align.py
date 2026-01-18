import array
import contextlib
import enum
import struct
def _Align(self, alignment):
    byte_width = 1 << alignment
    self._buf.extend(b'\x00' * _PaddingBytes(len(self._buf), byte_width))
    return byte_width