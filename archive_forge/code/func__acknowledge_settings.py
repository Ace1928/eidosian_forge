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
def _acknowledge_settings(self):
    """
        Acknowledge settings that have been received.

        .. versionchanged:: 2.0.0
           Removed from public API, removed useless ``event`` parameter, made
           automatic.

        :returns: Nothing
        """
    self.state_machine.process_input(ConnectionInputs.SEND_SETTINGS)
    changes = self.remote_settings.acknowledge()
    if SettingCodes.INITIAL_WINDOW_SIZE in changes:
        setting = changes[SettingCodes.INITIAL_WINDOW_SIZE]
        self._flow_control_change_from_settings(setting.original_value, setting.new_value)
    if SettingCodes.HEADER_TABLE_SIZE in changes:
        setting = changes[SettingCodes.HEADER_TABLE_SIZE]
        self.encoder.header_table_size = setting.new_value
    if SettingCodes.MAX_FRAME_SIZE in changes:
        setting = changes[SettingCodes.MAX_FRAME_SIZE]
        self.max_outbound_frame_size = setting.new_value
        for stream in self.streams.values():
            stream.max_outbound_frame_size = setting.new_value
    f = SettingsFrame(0)
    f.flags.add('ACK')
    return [f]