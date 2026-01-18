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
def _write_prefixed_body(self, bytes):
    self._write_func(b'b')
    self._write_func(struct.pack('!L', len(bytes)))
    self._write_func(bytes)