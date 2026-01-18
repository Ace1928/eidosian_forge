from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
def _read_more(self):
    next_read_size = self._protocol_decoder.next_read_size()
    if next_read_size == 0:
        self.finished_reading = True
        self._medium_request.finished_reading()
        return
    data = self._medium_request.read_bytes(next_read_size)
    if data == b'':
        if 'hpss' in debug.debug_flags:
            mutter('decoder state: buf[:10]=%r, state_accept=%s', self._protocol_decoder._get_in_buffer()[:10], self._protocol_decoder.state_accept.__name__)
        raise errors.ConnectionReset('Unexpected end of message. Please check connectivity and permissions, and report a bug if problems persist.')
    self._protocol_decoder.accept_bytes(data)