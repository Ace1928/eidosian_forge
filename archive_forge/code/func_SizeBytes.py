import array
import contextlib
import enum
import struct
@property
def SizeBytes(self):
    return self._buf[-self._byte_width:0]