import array
import contextlib
import enum
import struct
@staticmethod
def CompareKeys(a, b):
    if isinstance(a, Ref):
        a = a.AsKeyBytes
    if isinstance(b, Ref):
        b = b.AsKeyBytes
    return a < b