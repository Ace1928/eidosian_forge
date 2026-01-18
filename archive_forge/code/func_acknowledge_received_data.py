from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def acknowledge_received_data(self, acknowledged_size):
    """
        The user has informed us that they've processed some amount of data
        that was received on this stream. Pass that to the window manager and
        potentially return some WindowUpdate frames.
        """
    self.config.logger.debug('Acknowledge received data with size %d on %r', acknowledged_size, self)
    increment = self._inbound_window_manager.process_bytes(acknowledged_size)
    if increment:
        f = WindowUpdateFrame(self.stream_id)
        f.window_increment = increment
        return [f]
    return []