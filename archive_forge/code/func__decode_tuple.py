import _thread
import struct
import sys
import time
from collections import deque
from io import BytesIO
from fastbencode import bdecode_as_tuple, bencode
import breezy
from ... import debug, errors, osutils
from ...trace import log_exception_quietly, mutter
from . import message, request
def _decode_tuple(req_line):
    if req_line is None or req_line == b'':
        return None
    if not req_line.endswith(b'\n'):
        raise errors.SmartProtocolError('request %r not terminated' % req_line)
    return tuple(req_line[:-1].split(b'\x01'))