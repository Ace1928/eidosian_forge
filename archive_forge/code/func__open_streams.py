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
def _open_streams(self, remainder):
    """
        A common method of counting number of open streams. Returns the number
        of streams that are open *and* that have (stream ID % 2) == remainder.
        While it iterates, also deletes any closed streams.
        """
    count = 0
    to_delete = []
    for stream_id, stream in self.streams.items():
        if stream.open and stream_id % 2 == remainder:
            count += 1
        elif stream.closed:
            to_delete.append(stream_id)
    for stream_id in to_delete:
        stream = self.streams.pop(stream_id)
        self._closed_streams[stream_id] = stream.closed_by
    return count