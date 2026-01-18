from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class TextElements8(ValueField):
    string_textitem = Struct(LengthOf('string', 1), Int8('delta'), String8('string', pad=0))

    def pack_value(self, value):
        data = b''
        args = {}
        for v in value:
            if _PY3 and type(v) is str:
                v = v.encode('UTF-8')
            if type(v) is bytes:
                v = (0, v)
            if type(v) in (tuple, dict) or isinstance(v, DictWrapper):
                if type(v) is tuple:
                    delta, s = v
                else:
                    delta = v['delta']
                    s = v['string']
                while delta or s:
                    args['delta'] = delta
                    args['string'] = s[:254]
                    data = data + self.string_textitem.to_binary(*(), **args)
                    delta = 0
                    s = s[254:]
            else:
                if hasattr(v, '__fontable__'):
                    v = v.__fontable__()
                data = data + struct.pack('>BL', 255, v)
        dlen = len(data)
        return (data + b'\x00' * ((4 - dlen % 4) % 4), None, None)

    def parse_binary_value(self, data, display, length, format):
        values = []
        while 1:
            if len(data) < 2:
                break
            if _bytes_item(data[0]) == 255:
                values.append(struct.unpack('>L', data[1:5])[0])
                data = data[5:]
            elif _bytes_item(data[0]) == 0 and _bytes_item(data[1]) == 0:
                data = data[2:]
            else:
                v, data = self.string_textitem.parse_binary(data, display)
                values.append(v)
        return (values, b'')