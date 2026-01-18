from __future__ import annotations
import errno
import sys
from enum import Enum, IntEnum, IntFlag
class MessageOption(IntEnum):
    """Options on zmq.Frame objects

    .. versionadded:: 23
    """
    MORE = 1
    SHARED = 3
    SRCFD = 2