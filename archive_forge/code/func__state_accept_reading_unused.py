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
def _state_accept_reading_unused(self):
    self.unused_data += self._get_in_buffer()
    self._set_in_buffer(None)