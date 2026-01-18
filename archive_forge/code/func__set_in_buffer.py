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
def _set_in_buffer(self, new_buf):
    if new_buf is not None:
        if not isinstance(new_buf, bytes):
            raise TypeError(new_buf)
        self._in_buffer_list = [new_buf]
        self._in_buffer_len = len(new_buf)
    else:
        self._in_buffer_list = []
        self._in_buffer_len = 0