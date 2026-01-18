from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class ModifierMapping(ValueField):
    structcode = None

    def parse_binary_value(self, data, display, length, format):
        a = array(array_unsigned_codes[1], data[:8 * format])
        ret = []
        for i in range(0, 8):
            ret.append(a[i * format:(i + 1) * format])
        return (ret, data[8 * format:])

    def pack_value(self, value):
        if len(value) != 8:
            raise BadDataError('ModifierMapping list should have eight elements')
        keycodes = 0
        for v in value:
            keycodes = max(keycodes, len(v))
        a = array(array_unsigned_codes[1])
        for v in value:
            for k in v:
                a.append(k)
            for i in range(len(v), keycodes):
                a.append(0)
        return (a.tobytes(), len(value), keycodes)