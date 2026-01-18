from __future__ import annotations
import dataclasses
import enum
import io
import secrets
import struct
from typing import Callable, Generator, Optional, Sequence, Tuple
from . import exceptions, extensions
from .typing import Data
class CloseCode(enum.IntEnum):
    """Close code values for WebSocket close frames."""
    NORMAL_CLOSURE = 1000
    GOING_AWAY = 1001
    PROTOCOL_ERROR = 1002
    UNSUPPORTED_DATA = 1003
    NO_STATUS_RCVD = 1005
    ABNORMAL_CLOSURE = 1006
    INVALID_DATA = 1007
    POLICY_VIOLATION = 1008
    MESSAGE_TOO_BIG = 1009
    MANDATORY_EXTENSION = 1010
    INTERNAL_ERROR = 1011
    SERVICE_RESTART = 1012
    TRY_AGAIN_LATER = 1013
    BAD_GATEWAY = 1014
    TLS_HANDSHAKE = 1015