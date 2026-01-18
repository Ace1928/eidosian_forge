from __future__ import annotations
import socket
from abc import abstractmethod
from collections.abc import Callable, Collection, Mapping
from contextlib import AsyncExitStack
from io import IOBase
from ipaddress import IPv4Address, IPv6Address
from socket import AddressFamily
from types import TracebackType
from typing import Any, Tuple, TypeVar, Union
from .._core._typedattr import (
from ._streams import ByteStream, Listener, UnreliableObjectStream
from ._tasks import TaskGroup
@property
@abstractmethod
def _raw_socket(self) -> socket.socket:
    pass