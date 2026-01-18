import array
import contextlib
import enum
import struct
@property
def IsTypedVector(self):
    return Type.IsTypedVector(self._type)