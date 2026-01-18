from __future__ import annotations
import dataclasses
import enum
import io
import secrets
import struct
from typing import Callable, Generator, Optional, Sequence, Tuple
from . import exceptions, extensions
from .typing import Data
class Opcode(enum.IntEnum):
    """Opcode values for WebSocket frames."""
    CONT, TEXT, BINARY = (0, 1, 2)
    CLOSE, PING, PONG = (8, 9, 10)