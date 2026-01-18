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
def _receive_frame(self, frame):
    """
        Handle a frame received on the connection.

        .. versionchanged:: 2.0.0
           Removed from the public API.
        """
    try:
        frames, events = self._frame_dispatch_table[frame.__class__](frame)
    except StreamClosedError as e:
        if self._stream_is_closed_by_reset(e.stream_id):
            f = RstStreamFrame(e.stream_id)
            f.error_code = e.error_code
            self._prepare_for_sending([f])
            events = e._events
        else:
            raise
    except StreamIDTooLowError as e:
        if self._stream_is_closed_by_reset(e.stream_id):
            f = RstStreamFrame(e.stream_id)
            f.error_code = ErrorCodes.STREAM_CLOSED
            self._prepare_for_sending([f])
            events = []
        elif self._stream_is_closed_by_end(e.stream_id):
            raise StreamClosedError(e.stream_id)
        else:
            raise
    else:
        self._prepare_for_sending(frames)
    return events