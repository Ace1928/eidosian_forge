from __future__ import annotations
import errno
import sys
from enum import Enum, IntEnum, IntFlag
class NormMode(IntEnum):
    """Values for zmq.NORM_MODE socket option

    .. versionadded:: 26
    .. versionadded:: libzmq-4.3.5 (draft)
    """

    @staticmethod
    def _global_name(name):
        return f'NORM_{name}'
    FIXED = 0
    CC = 1
    CCL = 2
    CCE = 3
    CCE_ECNONLY = 4