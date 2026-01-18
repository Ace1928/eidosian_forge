import array
import contextlib
import enum
import struct
@property
def AsBool(self):
    if self._type is Type.BOOL:
        return bool(_Unpack(U, self._Bytes))
    else:
        return self.AsInt != 0