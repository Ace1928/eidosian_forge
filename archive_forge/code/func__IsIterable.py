import array
import contextlib
import enum
import struct
def _IsIterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False