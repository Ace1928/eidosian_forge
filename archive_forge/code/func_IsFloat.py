import array
import contextlib
import enum
import struct
@property
def IsFloat(self):
    return self._type in (Type.FLOAT, Type.INDIRECT_FLOAT)