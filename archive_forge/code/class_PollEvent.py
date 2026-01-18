from __future__ import annotations
import errno
import sys
from enum import Enum, IntEnum, IntFlag
class PollEvent(IntFlag):
    """Which events to poll for in poll methods

    .. versionadded: 23
    """
    POLLIN = 1
    POLLOUT = 2
    POLLERR = 4
    POLLPRI = 8