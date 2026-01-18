from __future__ import unicode_literals
import itertools
import struct
def _compat_bytes_to_byte_vals(byt):
    return [struct.unpack(b'!B', b)[0] for b in byt]