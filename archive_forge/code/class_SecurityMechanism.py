from __future__ import annotations
import errno
import sys
from enum import Enum, IntEnum, IntFlag
class SecurityMechanism(IntEnum):
    """Security mechanisms (as returned by ``socket.get(zmq.MECHANISM)``)

    .. versionadded:: 23
    """
    NULL = 0
    PLAIN = 1
    CURVE = 2
    GSSAPI = 3