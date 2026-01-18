from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class StrClass:
    structcode = None

    def pack_value(self, val):
        if type(val) is not bytes:
            val = val.encode('UTF-8')
        if _PY3:
            val = bytes([len(val)]) + val
        else:
            val = chr(len(val)) + val
        return val

    def parse_binary(self, data, display):
        slen = _bytes_item(data[0]) + 1
        s = data[1:slen]
        try:
            s = s.decode('UTF-8')
        except UnicodeDecodeError:
            pass
        return (s, data[slen:])