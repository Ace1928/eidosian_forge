import array
import contextlib
import enum
import struct
def _ReadKey(self, offset):
    key = self._buf[offset:]
    return key[:key.find(0)]