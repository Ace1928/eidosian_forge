import struct
from passlib import exc
from passlib.utils.compat import join_byte_values, byte_elem_value, \
def _unpack64(value):
    return _uint64_struct.unpack(value)[0]