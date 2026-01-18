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
def _get_or_create_stream(self, stream_id, allowed_ids):
    """
        Gets a stream by its stream ID. Will create one if one does not already
        exist. Use allowed_ids to circumvent the usual stream ID rules for
        clients and servers.

        .. versionchanged:: 2.0.0
           Removed this function from the public API.
        """
    try:
        return self.streams[stream_id]
    except KeyError:
        return self._begin_new_stream(stream_id, allowed_ids)