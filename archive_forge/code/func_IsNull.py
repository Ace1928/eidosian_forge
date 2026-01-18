import array
import contextlib
import enum
import struct
@property
def IsNull(self):
    return self._type is Type.NULL