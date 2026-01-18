import builtins
import codecs
import datetime
import select
import struct
import sys
import zlib
import subunit
import iso8601
def _encode_number(self, value):
    assert value >= 0
    if value < 64:
        return [struct.pack(FMT_8, value)]
    elif value < 16384:
        value = value | 16384
        return [struct.pack(FMT_16, value)]
    elif value < 4194304:
        value = value | 8388608
        return [struct.pack(FMT_16, value >> 8), struct.pack(FMT_8, value & 255)]
    elif value < 1073741824:
        value = value | 3221225472
        return [struct.pack(FMT_32, value)]
    else:
        raise ValueError('value too large to encode: %r' % (value,))