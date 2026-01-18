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
class _StatefulDecoder:
    """Base class for writing state machines to decode byte streams.

    Subclasses should provide a self.state_accept attribute that accepts bytes
    and, if appropriate, updates self.state_accept to a different function.
    accept_bytes will call state_accept as often as necessary to make sure the
    state machine has progressed as far as possible before it returns.

    See ProtocolThreeDecoder for an example subclass.
    """

    def __init__(self):
        self.finished_reading = False
        self._in_buffer_list = []
        self._in_buffer_len = 0
        self.unused_data = b''
        self.bytes_left = None
        self._number_needed_bytes = None

    def _get_in_buffer(self):
        if len(self._in_buffer_list) == 1:
            return self._in_buffer_list[0]
        in_buffer = b''.join(self._in_buffer_list)
        if len(in_buffer) != self._in_buffer_len:
            raise AssertionError('Length of buffer did not match expected value: %s != %s' % self._in_buffer_len, len(in_buffer))
        self._in_buffer_list = [in_buffer]
        return in_buffer

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

    def _set_in_buffer(self, new_buf):
        if new_buf is not None:
            if not isinstance(new_buf, bytes):
                raise TypeError(new_buf)
            self._in_buffer_list = [new_buf]
            self._in_buffer_len = len(new_buf)
        else:
            self._in_buffer_list = []
            self._in_buffer_len = 0

    def accept_bytes(self, new_buf):
        """Decode as much of bytes as possible.

        If 'new_buf' contains too much data it will be appended to
        self.unused_data.

        finished_reading will be set when no more data is required.  Further
        data will be appended to self.unused_data.
        """
        if not isinstance(new_buf, bytes):
            raise TypeError(new_buf)
        self._number_needed_bytes = None
        self._in_buffer_list.append(new_buf)
        self._in_buffer_len += len(new_buf)
        try:
            current_state = self.state_accept
            self.state_accept()
            while current_state != self.state_accept:
                current_state = self.state_accept
                self.state_accept()
        except _NeedMoreBytes as e:
            self._number_needed_bytes = e.count