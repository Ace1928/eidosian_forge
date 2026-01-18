import array
import contextlib
import enum
import struct
@property
def IsKey(self):
    return self._type is Type.KEY