import struct
from passlib import exc
from passlib.utils.compat import join_byte_values, byte_elem_value, \
def _pack56(value):
    return _uint64_struct.pack(value)[1:]