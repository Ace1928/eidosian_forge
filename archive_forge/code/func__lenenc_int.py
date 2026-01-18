import errno
import os
import socket
import struct
import sys
import traceback
import warnings
from . import _auth
from .charset import charset_by_name, charset_by_id
from .constants import CLIENT, COMMAND, CR, ER, FIELD_TYPE, SERVER_STATUS
from . import converters
from .cursors import Cursor
from .optionfile import Parser
from .protocol import (
from . import err, VERSION_STRING
def _lenenc_int(i):
    if i < 0:
        raise ValueError('Encoding %d is less than 0 - no representation in LengthEncodedInteger' % i)
    elif i < 251:
        return bytes([i])
    elif i < 1 << 16:
        return b'\xfc' + struct.pack('<H', i)
    elif i < 1 << 24:
        return b'\xfd' + struct.pack('<I', i)[:3]
    elif i < 1 << 64:
        return b'\xfe' + struct.pack('<Q', i)
    else:
        raise ValueError('Encoding %x is larger than %x - no representation in LengthEncodedInteger' % (i, 1 << 64))