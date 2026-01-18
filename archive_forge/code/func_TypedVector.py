import array
import contextlib
import enum
import struct
@contextlib.contextmanager
def TypedVector(self, key=None):
    if key:
        self.Key(key)
    try:
        start = self._StartVector()
        yield self
    finally:
        self._EndVector(start, typed=True, fixed=False)