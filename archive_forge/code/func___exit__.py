from __future__ import annotations
import contextlib
import logging
import random
import socket
import struct
import threading
import uuid
from types import TracebackType
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Type, Union
from ..exceptions import ConnectionClosed, ConnectionClosedOK, ProtocolError
from ..frames import DATA_OPCODES, BytesLike, CloseCode, Frame, Opcode, prepare_ctrl
from ..http11 import Request, Response
from ..protocol import CLOSED, OPEN, Event, Protocol, State
from ..typing import Data, LoggerLike, Subprotocol
from .messages import Assembler
from .utils import Deadline
def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
    if exc_type is None:
        self.close()
    else:
        self.close(CloseCode.INTERNAL_ERROR)