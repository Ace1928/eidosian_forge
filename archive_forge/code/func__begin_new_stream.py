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
def _begin_new_stream(self, stream_id, allowed_ids):
    """
        Initiate a new stream.

        .. versionchanged:: 2.0.0
           Removed this function from the public API.

        :param stream_id: The ID of the stream to open.
        :param allowed_ids: What kind of stream ID is allowed.
        """
    self.config.logger.debug('Attempting to initiate stream ID %d', stream_id)
    outbound = self._stream_id_is_outbound(stream_id)
    highest_stream_id = self.highest_outbound_stream_id if outbound else self.highest_inbound_stream_id
    if stream_id <= highest_stream_id:
        raise StreamIDTooLowError(stream_id, highest_stream_id)
    if stream_id % 2 != int(allowed_ids):
        raise ProtocolError('Invalid stream ID for peer.')
    s = H2Stream(stream_id, config=self.config, inbound_window_size=self.local_settings.initial_window_size, outbound_window_size=self.remote_settings.initial_window_size)
    self.config.logger.debug('Stream ID %d created', stream_id)
    s.max_inbound_frame_size = self.max_inbound_frame_size
    s.max_outbound_frame_size = self.max_outbound_frame_size
    self.streams[stream_id] = s
    self.config.logger.debug('Current streams: %s', self.streams.keys())
    if outbound:
        self.highest_outbound_stream_id = stream_id
    else:
        self.highest_inbound_stream_id = stream_id
    return s