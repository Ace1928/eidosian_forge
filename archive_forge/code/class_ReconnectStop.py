from __future__ import annotations
import errno
import sys
from enum import Enum, IntEnum, IntFlag
class ReconnectStop(IntEnum):
    """Select behavior for socket.reconnect_stop

    .. versionadded:: 25
    """

    @staticmethod
    def _global_name(name):
        return f'RECONNECT_STOP_{name}'
    CONN_REFUSED = 1
    HANDSHAKE_FAILED = 2
    AFTER_DISCONNECT = 4