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
def _receive_ping_frame(self, frame):
    """
        Receive a PING frame on the connection.
        """
    events = self.state_machine.process_input(ConnectionInputs.RECV_PING)
    flags = []
    if 'ACK' in frame.flags:
        evt = PingAckReceived()
    else:
        evt = PingReceived()
        f = PingFrame(0)
        f.flags = {'ACK'}
        f.opaque_data = frame.opaque_data
        flags.append(f)
    evt.ping_data = frame.opaque_data
    events.append(evt)
    return (flags, events)