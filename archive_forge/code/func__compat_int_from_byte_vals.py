from __future__ import unicode_literals
import itertools
import struct
def _compat_int_from_byte_vals(bytvals, endianess):
    assert endianess == 'big'
    res = 0
    for bv in bytvals:
        assert isinstance(bv, _compat_int_types)
        res = (res << 8) + bv
    return res