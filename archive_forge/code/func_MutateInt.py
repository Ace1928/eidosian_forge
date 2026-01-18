import array
import contextlib
import enum
import struct
def MutateInt(self, value):
    """Mutates underlying integer value bytes in place.

    Args:
      value: New integer value. It must fit to the byte size of the existing
        encoded value.

    Returns:
      Whether the value was mutated or not.
    """
    if self._type is Type.INT:
        return _Mutate(I, self._buf, value, self._parent_width, BitWidth.I(value))
    elif self._type is Type.INDIRECT_INT:
        return _Mutate(I, self._Indirect(), value, self._byte_width, BitWidth.I(value))
    elif self._type is Type.UINT:
        return _Mutate(U, self._buf, value, self._parent_width, BitWidth.U(value))
    elif self._type is Type.INDIRECT_UINT:
        return _Mutate(U, self._Indirect(), value, self._byte_width, BitWidth.U(value))
    else:
        return False