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
class LengthPrefixedBodyDecoder(_StatefulDecoder):
    """Decodes the length-prefixed bulk data."""

    def __init__(self):
        _StatefulDecoder.__init__(self)
        self.state_accept = self._state_accept_expecting_length
        self.state_read = self._state_read_no_data
        self._body = b''
        self._trailer_buffer = b''

    def next_read_size(self):
        if self.bytes_left is not None:
            return self.bytes_left + 5
        elif self.state_accept == self._state_accept_reading_trailer:
            return 5 - len(self._trailer_buffer)
        elif self.state_accept == self._state_accept_expecting_length:
            return 6
        else:
            return 1

    def read_pending_data(self):
        """Return any pending data that has been decoded."""
        return self.state_read()

    def _state_accept_expecting_length(self):
        in_buf = self._get_in_buffer()
        pos = in_buf.find(b'\n')
        if pos == -1:
            return
        self.bytes_left = int(in_buf[:pos])
        self._set_in_buffer(in_buf[pos + 1:])
        self.state_accept = self._state_accept_reading_body
        self.state_read = self._state_read_body_buffer

    def _state_accept_reading_body(self):
        in_buf = self._get_in_buffer()
        self._body += in_buf
        self.bytes_left -= len(in_buf)
        self._set_in_buffer(None)
        if self.bytes_left <= 0:
            if self.bytes_left != 0:
                self._trailer_buffer = self._body[self.bytes_left:]
                self._body = self._body[:self.bytes_left]
            self.bytes_left = None
            self.state_accept = self._state_accept_reading_trailer

    def _state_accept_reading_trailer(self):
        self._trailer_buffer += self._get_in_buffer()
        self._set_in_buffer(None)
        if self._trailer_buffer.startswith(b'done\n'):
            self.unused_data = self._trailer_buffer[len(b'done\n'):]
            self.state_accept = self._state_accept_reading_unused
            self.finished_reading = True

    def _state_accept_reading_unused(self):
        self.unused_data += self._get_in_buffer()
        self._set_in_buffer(None)

    def _state_read_no_data(self):
        return b''

    def _state_read_body_buffer(self):
        result = self._body
        self._body = b''
        return result