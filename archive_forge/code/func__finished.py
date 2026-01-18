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
def _finished(self):
    self.unused_data = self._get_in_buffer()
    self._in_buffer_list = []
    self._in_buffer_len = 0
    self.state_accept = self._state_accept_reading_unused
    if self.error:
        error_args = tuple(self.error_in_progress)
        self.chunks.append(request.FailedSmartServerResponse(error_args))
        self.error_in_progress = None
    self.finished_reading = True