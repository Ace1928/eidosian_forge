from __future__ import annotations
import atexit
import os
from threading import Lock
from typing import Any, Callable, Generic, TypeVar, overload
from warnings import warn
from weakref import WeakSet
import zmq
from zmq._typing import TypeAlias
from zmq.backend import Context as ContextBase
from zmq.constants import ContextOption, Errno, SocketOption
from zmq.error import ZMQError
from zmq.utils.interop import cast_int_addr
from .attrsettr import AttributeSetter, OptValT
from .socket import Socket, SyncSocket
def _notice_atexit() -> None:
    global _exiting
    _exiting = True