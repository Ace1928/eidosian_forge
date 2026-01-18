from __future__ import annotations
import enum
import logging
import uuid
from typing import Generator, List, Optional, Type, Union
from .exceptions import (
from .extensions import Extension
from .frames import (
from .http11 import Request, Response
from .streams import StreamReader
from .typing import LoggerLike, Origin, Subprotocol
class Side(enum.IntEnum):
    """A WebSocket connection is either a server or a client."""
    SERVER, CLIENT = range(2)