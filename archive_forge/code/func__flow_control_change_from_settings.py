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
def _flow_control_change_from_settings(self, old_value, new_value):
    """
        Update flow control windows in response to a change in the value of
        SETTINGS_INITIAL_WINDOW_SIZE.

        When this setting is changed, it automatically updates all flow control
        windows by the delta in the settings values. Note that it does not
        increment the *connection* flow control window, per section 6.9.2 of
        RFC 7540.
        """
    delta = new_value - old_value
    for stream in self.streams.values():
        stream.outbound_flow_control_window = guard_increment_window(stream.outbound_flow_control_window, delta)