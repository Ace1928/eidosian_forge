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
def _state_accept_expecting_headers(self):
    decoded = self._extract_prefixed_bencoded_data()
    if not isinstance(decoded, dict):
        raise errors.SmartProtocolError('Header object {!r} is not a dict'.format(decoded))
    self.state_accept = self._state_accept_expecting_message_part
    try:
        self.message_handler.headers_received(decoded)
    except:
        raise SmartMessageHandlerError(sys.exc_info())