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
def _receive_window_update_frame(self, frame):
    """
        Receive a WINDOW_UPDATE frame on the connection.
        """
    if not 1 <= frame.window_increment <= self.MAX_WINDOW_INCREMENT:
        raise ProtocolError('Flow control increment must be between 1 and %d, received %d' % (self.MAX_WINDOW_INCREMENT, frame.window_increment))
    events = self.state_machine.process_input(ConnectionInputs.RECV_WINDOW_UPDATE)
    if frame.stream_id:
        stream = self._get_stream_by_id(frame.stream_id)
        frames, stream_events = stream.receive_window_update(frame.window_increment)
    else:
        self.outbound_flow_control_window = guard_increment_window(self.outbound_flow_control_window, frame.window_increment)
        window_updated_event = WindowUpdated()
        window_updated_event.stream_id = 0
        window_updated_event.delta = frame.window_increment
        stream_events = [window_updated_event]
        frames = []
    return (frames, events + stream_events)