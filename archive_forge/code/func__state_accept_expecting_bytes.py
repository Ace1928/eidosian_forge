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
def _state_accept_expecting_bytes(self):
    prefixed_bytes = self._extract_length_prefixed_bytes()
    self.state_accept = self._state_accept_expecting_message_part
    try:
        self.message_handler.bytes_part_received(prefixed_bytes)
    except:
        raise SmartMessageHandlerError(sys.exc_info())