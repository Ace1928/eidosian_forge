from __future__ import annotations
import errno
import sys
from enum import Enum, IntEnum, IntFlag
class RouterNotify(IntEnum):
    """Values for zmq.ROUTER_NOTIFY socket option

    .. versionadded:: 26
    .. versionadded:: libzmq-4.3.0 (draft)
    """

    @staticmethod
    def _global_name(name):
        return f'NOTIFY_{name}'
    CONNECT = 1
    DISCONNECT = 2