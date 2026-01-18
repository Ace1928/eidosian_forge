import array
import contextlib
import enum
import struct
@property
def AsFloat(self):
    """Returns current reference as floating point value."""
    if self.IsNull:
        return 0.0
    elif self.IsBool:
        return float(self.AsBool)
    elif self.IsInt:
        return float(self.AsInt)
    elif self._type is Type.FLOAT:
        return _Unpack(F, self._Bytes)
    elif self._type is Type.INDIRECT_FLOAT:
        return _Unpack(F, self._Indirect()[:self._byte_width])
    elif self.IsString:
        return float(self.AsString)
    elif self.IsVector:
        return float(len(self.AsVector))
    elif self.IsTypedVector():
        return float(len(self.AsTypedVector))
    elif self.IsFixedTypedVector():
        return float(len(self.FixedTypedVector))
    else:
        raise self._ConvertError(Type.FLOAT)