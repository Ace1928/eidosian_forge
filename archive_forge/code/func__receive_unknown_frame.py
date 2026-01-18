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
def _receive_unknown_frame(self, frame):
    """
        We have received a frame that we do not understand. This is almost
        certainly an extension frame, though it's impossible to be entirely
        sure.

        RFC 7540 § 5.5 says that we MUST ignore unknown frame types: so we
        do. We do notify the user that we received one, however.
        """
    self.config.logger.debug('Received unknown extension frame (ID %d)', frame.stream_id)
    event = UnknownFrameReceived()
    event.frame = frame
    return ([], [event])