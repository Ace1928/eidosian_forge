import array
import contextlib
import enum
import struct
@property
def AsString(self):
    if self.IsString:
        return str(String(self._Indirect(), self._byte_width))
    elif self.IsKey:
        return self.AsKey
    else:
        raise self._ConvertError(Type.STRING)