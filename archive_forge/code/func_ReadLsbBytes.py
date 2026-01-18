from __future__ import division
import os
import struct
from pyu2f import errors
from pyu2f.hid import base
def ReadLsbBytes(rd, offset, value_size):
    """Reads value_size bytes from rd at offset, least signifcant byte first."""
    encoding = None
    if value_size == 1:
        encoding = '<B'
    elif value_size == 2:
        encoding = '<H'
    elif value_size == 4:
        encoding = '<L'
    else:
        raise errors.HidError('Invalid value size specified')
    ret, = struct.unpack(encoding, rd[offset:offset + value_size])
    return ret