import array
import contextlib
import enum
import struct
def Null(self, key=None):
    """Encodes None value."""
    if key:
        self.Key(key)
    self._stack.append(Value.Null())