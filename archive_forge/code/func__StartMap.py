import array
import contextlib
import enum
import struct
def _StartMap(self):
    """Starts map construction."""
    return len(self._stack)