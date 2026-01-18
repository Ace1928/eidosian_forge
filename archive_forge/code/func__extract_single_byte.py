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
def _extract_single_byte(self):
    if self._in_buffer_len == 0:
        raise _NeedMoreBytes(1)
    in_buf = self._get_in_buffer()
    one_byte = in_buf[0:1]
    self._set_in_buffer(in_buf[1:])
    return one_byte