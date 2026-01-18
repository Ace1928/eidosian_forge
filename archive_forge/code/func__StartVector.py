import array
import contextlib
import enum
import struct
def _StartVector(self):
    """Starts vector construction."""
    return len(self._stack)