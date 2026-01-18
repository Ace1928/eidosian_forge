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
def _extract_line(self):
    in_buf = self._get_in_buffer()
    pos = in_buf.find(b'\n')
    if pos == -1:
        raise _NeedMoreBytes(1)
    line = in_buf[:pos]
    self._set_in_buffer(in_buf[pos + 1:])
    return line