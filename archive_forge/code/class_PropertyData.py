from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class PropertyData(ValueField):
    structcode = None

    def parse_binary_value(self, data, display, length, format):
        if length is None:
            length = len(data) // (format // 8)
        else:
            length = int(length)
        if format == 0:
            ret = None
            return (ret, data)
        elif format == 8:
            ret = (8, data[:length])
            data = data[length + (4 - length % 4) % 4:]
        elif format == 16:
            ret = (16, array(array_unsigned_codes[2], data[:2 * length]))
            data = data[2 * (length + length % 2):]
        elif format == 32:
            ret = (32, array(array_unsigned_codes[4], data[:4 * length]))
            data = data[4 * length:]
        if type(ret[1]) is bytes:
            try:
                ret = (ret[0], ret[1].decode('UTF-8'))
            except UnicodeDecodeError:
                pass
        return (ret, data)

    def pack_value(self, value):
        fmt, val = value
        if fmt not in (8, 16, 32):
            raise BadDataError('Invalid property data format %d' % fmt)
        if _PY3 and type(val) is str:
            val = val.encode('UTF-8')
        if type(val) is bytes:
            size = fmt // 8
            vlen = len(val)
            if vlen % size:
                vlen = vlen - vlen % size
                data = val[:vlen]
            else:
                data = val
            dlen = vlen // size
        else:
            if type(val) is tuple:
                val = list(val)
            size = fmt // 8
            data = array(array_unsigned_codes[size], val).tobytes()
            dlen = len(val)
        dl = len(data)
        data = data + b'\x00' * ((4 - dl % 4) % 4)
        return (data, dlen, fmt)