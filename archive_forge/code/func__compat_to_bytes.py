from __future__ import unicode_literals
import itertools
import struct
def _compat_to_bytes(intval, length, endianess):
    assert isinstance(intval, _compat_int_types)
    assert endianess == 'big'
    if length == 4:
        if intval < 0 or intval >= 2 ** 32:
            raise struct.error("integer out of range for 'I' format code")
        return struct.pack(b'!I', intval)
    elif length == 16:
        if intval < 0 or intval >= 2 ** 128:
            raise struct.error("integer out of range for 'QQ' format code")
        return struct.pack(b'!QQ', intval >> 64, intval & 18446744073709551615)
    else:
        raise NotImplementedError()