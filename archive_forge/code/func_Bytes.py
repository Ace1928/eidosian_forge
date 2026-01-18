import array
import contextlib
import enum
import struct
@property
def Bytes(self):
    return self._buf[:self._byte_width * len(self)]