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
def _get_in_bytes(self, count):
    """Grab X bytes from the input_buffer.

        Callers should have already checked that self._in_buffer_len is >
        count. Note, this does not consume the bytes from the buffer. The
        caller will still need to call _get_in_buffer() and then
        _set_in_buffer() if they actually need to consume the bytes.
        """
    if len(self._in_buffer_list) == 0:
        raise AssertionError('Callers must be sure we have buffered bytes before calling _get_in_bytes')
    if len(self._in_buffer_list[0]) > count:
        return self._in_buffer_list[0][:count]
    in_buf = self._get_in_buffer()
    return in_buf[:count]