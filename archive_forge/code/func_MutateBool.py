import array
import contextlib
import enum
import struct
def MutateBool(self, value):
    """Mutates underlying boolean value bytes in place.

    Args:
      value: New boolean value.

    Returns:
      Whether the value was mutated or not.
    """
    return self.IsBool and _Mutate(U, self._buf, value, self._parent_width, BitWidth.W8)