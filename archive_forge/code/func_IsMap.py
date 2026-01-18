import array
import contextlib
import enum
import struct
@property
def IsMap(self):
    return self._type is Type.MAP