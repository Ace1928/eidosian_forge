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
class H2ConnectionStateMachine(object):
    """
    A single HTTP/2 connection state machine.

    This state machine, while defined in its own class, is logically part of
    the H2Connection class also defined in this file. The state machine itself
    maintains very little state directly, instead focusing entirely on managing
    state transitions.
    """
    _transitions = {(ConnectionState.IDLE, ConnectionInputs.SEND_HEADERS): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.IDLE, ConnectionInputs.RECV_HEADERS): (None, ConnectionState.SERVER_OPEN), (ConnectionState.IDLE, ConnectionInputs.SEND_SETTINGS): (None, ConnectionState.IDLE), (ConnectionState.IDLE, ConnectionInputs.RECV_SETTINGS): (None, ConnectionState.IDLE), (ConnectionState.IDLE, ConnectionInputs.SEND_WINDOW_UPDATE): (None, ConnectionState.IDLE), (ConnectionState.IDLE, ConnectionInputs.RECV_WINDOW_UPDATE): (None, ConnectionState.IDLE), (ConnectionState.IDLE, ConnectionInputs.SEND_PING): (None, ConnectionState.IDLE), (ConnectionState.IDLE, ConnectionInputs.RECV_PING): (None, ConnectionState.IDLE), (ConnectionState.IDLE, ConnectionInputs.SEND_GOAWAY): (None, ConnectionState.CLOSED), (ConnectionState.IDLE, ConnectionInputs.RECV_GOAWAY): (None, ConnectionState.CLOSED), (ConnectionState.IDLE, ConnectionInputs.SEND_PRIORITY): (None, ConnectionState.IDLE), (ConnectionState.IDLE, ConnectionInputs.RECV_PRIORITY): (None, ConnectionState.IDLE), (ConnectionState.IDLE, ConnectionInputs.SEND_ALTERNATIVE_SERVICE): (None, ConnectionState.SERVER_OPEN), (ConnectionState.IDLE, ConnectionInputs.RECV_ALTERNATIVE_SERVICE): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.SEND_HEADERS): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.SEND_DATA): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.SEND_GOAWAY): (None, ConnectionState.CLOSED), (ConnectionState.CLIENT_OPEN, ConnectionInputs.SEND_WINDOW_UPDATE): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.SEND_PING): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.SEND_SETTINGS): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.SEND_PRIORITY): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.RECV_HEADERS): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.RECV_PUSH_PROMISE): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.RECV_DATA): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.RECV_GOAWAY): (None, ConnectionState.CLOSED), (ConnectionState.CLIENT_OPEN, ConnectionInputs.RECV_WINDOW_UPDATE): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.RECV_PING): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.RECV_SETTINGS): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.SEND_RST_STREAM): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.RECV_RST_STREAM): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.RECV_PRIORITY): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.CLIENT_OPEN, ConnectionInputs.RECV_ALTERNATIVE_SERVICE): (None, ConnectionState.CLIENT_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.SEND_HEADERS): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.SEND_PUSH_PROMISE): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.SEND_DATA): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.SEND_GOAWAY): (None, ConnectionState.CLOSED), (ConnectionState.SERVER_OPEN, ConnectionInputs.SEND_WINDOW_UPDATE): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.SEND_PING): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.SEND_SETTINGS): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.SEND_PRIORITY): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.RECV_HEADERS): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.RECV_DATA): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.RECV_GOAWAY): (None, ConnectionState.CLOSED), (ConnectionState.SERVER_OPEN, ConnectionInputs.RECV_WINDOW_UPDATE): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.RECV_PING): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.RECV_SETTINGS): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.RECV_PRIORITY): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.SEND_RST_STREAM): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.RECV_RST_STREAM): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.SEND_ALTERNATIVE_SERVICE): (None, ConnectionState.SERVER_OPEN), (ConnectionState.SERVER_OPEN, ConnectionInputs.RECV_ALTERNATIVE_SERVICE): (None, ConnectionState.SERVER_OPEN), (ConnectionState.CLOSED, ConnectionInputs.SEND_GOAWAY): (None, ConnectionState.CLOSED), (ConnectionState.CLOSED, ConnectionInputs.RECV_GOAWAY): (None, ConnectionState.CLOSED)}

    def __init__(self):
        self.state = ConnectionState.IDLE

    def process_input(self, input_):
        """
        Process a specific input in the state machine.
        """
        if not isinstance(input_, ConnectionInputs):
            raise ValueError('Input must be an instance of ConnectionInputs')
        try:
            func, target_state = self._transitions[self.state, input_]
        except KeyError:
            old_state = self.state
            self.state = ConnectionState.CLOSED
            raise ProtocolError('Invalid input %s in state %s' % (input_, old_state))
        else:
            self.state = target_state
            if func is not None:
                return func()
            return []