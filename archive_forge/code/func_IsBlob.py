import array
import contextlib
import enum
import struct
@property
def IsBlob(self):
    return self._type is Type.BLOB