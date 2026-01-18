import array
import contextlib
import enum
import struct
def _WriteVector(self, fmt, values, byte_width):
    self._buf.extend(_PackVector(fmt, values, byte_width))