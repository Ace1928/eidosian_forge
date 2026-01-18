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
def _serialise_offsets(self, offsets):
    """Serialise a readv offset list."""
    txt = []
    for start, length in offsets:
        txt.append(b'%d,%d' % (start, length))
    return b'\n'.join(txt)