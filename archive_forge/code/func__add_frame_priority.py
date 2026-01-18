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
def _add_frame_priority(frame, weight=None, depends_on=None, exclusive=None):
    """
    Adds priority data to a given frame. Does not change any flags set on that
    frame: if the caller is adding priority information to a HEADERS frame they
    must set that themselves.

    This method also deliberately sets defaults for anything missing.

    This method validates the input values.
    """
    if depends_on == frame.stream_id:
        raise ProtocolError('Stream %d may not depend on itself' % frame.stream_id)
    if weight is not None:
        if weight > 256 or weight < 1:
            raise ProtocolError('Weight must be between 1 and 256, not %d' % weight)
        else:
            weight -= 1
    weight = weight if weight is not None else 15
    depends_on = depends_on if depends_on is not None else 0
    exclusive = exclusive if exclusive is not None else False
    frame.stream_weight = weight
    frame.depends_on = depends_on
    frame.exclusive = exclusive
    return frame