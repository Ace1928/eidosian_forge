from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class ValueList(Field):
    structcode = None
    keyword_args = 1
    default = 'usekeywords'

    def __init__(self, name, mask, pad, *fields):
        self.name = name
        self.maskcode = '=%s%dx' % (unsigned_codes[mask], pad)
        self.maskcodelen = struct.calcsize(self.maskcode)
        self.fields = []
        flag = 1
        for f in fields:
            if f.name:
                self.fields.append((f, flag))
                flag = flag << 1

    def pack_value(self, arg, keys):
        mask = 0
        data = b''
        if arg == self.default:
            arg = keys
        for field, flag in self.fields:
            if field.name in arg:
                mask = mask | flag
                val = arg[field.name]
                if field.check_value is not None:
                    val = field.check_value(val)
                d = struct.pack('=' + field.structcode, val)
                data = data + d + b'\x00' * (4 - len(d))
        return (struct.pack(self.maskcode, mask) + data, None, None)

    def parse_binary_value(self, data, display, length, format):
        r = {}
        mask = int(struct.unpack(self.maskcode, data[:self.maskcodelen])[0])
        data = data[self.maskcodelen:]
        for field, flag in self.fields:
            if mask & flag:
                if field.structcode:
                    vals = struct.unpack('=' + field.structcode, data[:struct.calcsize('=' + field.structcode)])
                    if field.structvalues == 1:
                        vals = vals[0]
                    if field.parse_value is not None:
                        vals = field.parse_value(vals, display)
                else:
                    vals, d = field.parse_binary_value(data[:4], display, None, None)
                r[field.name] = vals
                data = data[4:]
        return (DictWrapper(r), data)