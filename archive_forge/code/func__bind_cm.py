from __future__ import annotations
import errno
import pickle
import random
import sys
from typing import (
from warnings import warn
import zmq
from zmq._typing import Literal, TypeAlias
from zmq.backend import Socket as SocketBase
from zmq.error import ZMQBindError, ZMQError
from zmq.utils import jsonapi
from zmq.utils.interop import cast_int_addr
from ..constants import SocketOption, SocketType, _OptType
from .attrsettr import AttributeSetter
from .poll import Poller
def _bind_cm(self: _SocketType, addr: str) -> _SocketContext[_SocketType]:
    """Context manager to unbind on exit

        .. versionadded:: 20.0
        """
    try:
        addr = cast(bytes, self.get(zmq.LAST_ENDPOINT)).decode('utf8')
    except (AttributeError, ZMQError, UnicodeDecodeError):
        pass
    return _SocketContext(self, 'bind', addr)