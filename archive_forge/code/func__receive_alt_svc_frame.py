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
def _receive_alt_svc_frame(self, frame):
    """
        An ALTSVC frame has been received. This frame, specified in RFC 7838,
        is used to advertise alternative places where the same service can be
        reached.

        This frame can optionally be received either on a stream or on stream
        0, and its semantics are different in each case.
        """
    events = self.state_machine.process_input(ConnectionInputs.RECV_ALTERNATIVE_SERVICE)
    frames = []
    if frame.stream_id:
        try:
            stream = self._get_stream_by_id(frame.stream_id)
        except (NoSuchStreamError, StreamClosedError):
            pass
        else:
            stream_frames, stream_events = stream.receive_alt_svc(frame)
            frames.extend(stream_frames)
            events.extend(stream_events)
    else:
        if not frame.origin:
            return (frames, events)
        if not self.config.client_side:
            return (frames, events)
        event = AlternativeServiceAvailable()
        event.origin = frame.origin
        event.field_value = frame.field
        events.append(event)
    return (frames, events)