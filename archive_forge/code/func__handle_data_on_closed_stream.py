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
def _handle_data_on_closed_stream(self, events, exc, frame):
    frames = []
    conn_manager = self._inbound_flow_control_window_manager
    conn_increment = conn_manager.process_bytes(frame.flow_controlled_length)
    if conn_increment:
        f = WindowUpdateFrame(0)
        f.window_increment = conn_increment
        frames.append(f)
        self.config.logger.debug('Received DATA frame on closed stream %d - auto-emitted a WINDOW_UPDATE by %d', frame.stream_id, conn_increment)
    f = RstStreamFrame(exc.stream_id)
    f.error_code = exc.error_code
    frames.append(f)
    self.config.logger.debug('Stream %d already CLOSED or cleaned up - auto-emitted a RST_FRAME' % frame.stream_id)
    return (frames, events + exc._events)