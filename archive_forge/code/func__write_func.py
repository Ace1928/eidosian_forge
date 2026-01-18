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
def _write_func(self, bytes):
    self._buf.append(bytes)
    self._buf_len += len(bytes)
    if self._buf_len > self.BUFFER_SIZE:
        self.flush()