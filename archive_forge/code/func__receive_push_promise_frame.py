import base64
from enum import Enum, IntEnum
from hyperframe.exceptions import InvalidPaddingError
from hyperframe.frame import (
from hpack.hpack import Encoder, Decoder
from hpack.exceptions import HPACKError, OversizedHeaderListError
from .config import H2Configuration
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .frame_buffer import FrameBuffer
from .settings import Settings, SettingCodes
from .stream import H2Stream, StreamClosedBy
from .utilities import SizeLimitDict, guard_increment_window
from .windows import WindowManager
def _receive_push_promise_frame(self, frame):
    """
        Receive a push-promise frame on the connection.
        """
    if not self.local_settings.enable_push:
        raise ProtocolError('Received pushed stream')
    pushed_headers = _decode_headers(self.decoder, frame.data)
    events = self.state_machine.process_input(ConnectionInputs.RECV_PUSH_PROMISE)
    try:
        stream = self._get_stream_by_id(frame.stream_id)
    except NoSuchStreamError:
        if self._stream_closed_by(frame.stream_id) == StreamClosedBy.SEND_RST_STREAM:
            f = RstStreamFrame(frame.promised_stream_id)
            f.error_code = ErrorCodes.REFUSED_STREAM
            return ([f], events)
        raise ProtocolError('Attempted to push on closed stream.')
    if frame.stream_id % 2 == 0:
        raise ProtocolError('Cannot recursively push streams.')
    try:
        frames, stream_events = stream.receive_push_promise_in_band(frame.promised_stream_id, pushed_headers, self.config.header_encoding)
    except StreamClosedError:
        f = RstStreamFrame(frame.promised_stream_id)
        f.error_code = ErrorCodes.REFUSED_STREAM
        return ([f], events)
    new_stream = self._begin_new_stream(frame.promised_stream_id, AllowedStreamIDs.EVEN)
    self.streams[frame.promised_stream_id] = new_stream
    new_stream.remotely_pushed(pushed_headers)
    return (frames, events + stream_events)