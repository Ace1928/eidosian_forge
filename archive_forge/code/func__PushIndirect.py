import array
import contextlib
import enum
import struct
def _PushIndirect(self, value, type_, bit_width):
    byte_width = self._Align(bit_width)
    loc = len(self._buf)
    fmt = {Type.INDIRECT_INT: I, Type.INDIRECT_UINT: U, Type.INDIRECT_FLOAT: F}[type_]
    self._Write(fmt, value, byte_width)
    self._stack.append(Value(loc, type_, bit_width))