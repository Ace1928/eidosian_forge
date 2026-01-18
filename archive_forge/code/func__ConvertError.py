import array
import contextlib
import enum
import struct
def _ConvertError(self, target_type):
    raise TypeError('cannot convert %s to %s' % (self._type, target_type))