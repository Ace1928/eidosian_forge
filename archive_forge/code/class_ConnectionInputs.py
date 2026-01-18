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
class ConnectionInputs(Enum):
    SEND_HEADERS = 0
    SEND_PUSH_PROMISE = 1
    SEND_DATA = 2
    SEND_GOAWAY = 3
    SEND_WINDOW_UPDATE = 4
    SEND_PING = 5
    SEND_SETTINGS = 6
    SEND_RST_STREAM = 7
    SEND_PRIORITY = 8
    RECV_HEADERS = 9
    RECV_PUSH_PROMISE = 10
    RECV_DATA = 11
    RECV_GOAWAY = 12
    RECV_WINDOW_UPDATE = 13
    RECV_PING = 14
    RECV_SETTINGS = 15
    RECV_RST_STREAM = 16
    RECV_PRIORITY = 17
    SEND_ALTERNATIVE_SERVICE = 18
    RECV_ALTERNATIVE_SERVICE = 19