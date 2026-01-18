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
def _state_accept_reading_trailer(self):
    self._trailer_buffer += self._get_in_buffer()
    self._set_in_buffer(None)
    if self._trailer_buffer.startswith(b'done\n'):
        self.unused_data = self._trailer_buffer[len(b'done\n'):]
        self.state_accept = self._state_accept_reading_unused
        self.finished_reading = True