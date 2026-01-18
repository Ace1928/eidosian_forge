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
class UNIXSocketStream(SocketStream):

    @abstractmethod
    async def send_fds(self, message: bytes, fds: Collection[int | IOBase]) -> None:
        """
        Send file descriptors along with a message to the peer.

        :param message: a non-empty bytestring
        :param fds: a collection of files (either numeric file descriptors or open file
            or socket objects)
        """

    @abstractmethod
    async def receive_fds(self, msglen: int, maxfds: int) -> tuple[bytes, list[int]]:
        """
        Receive file descriptors along with a message from the peer.

        :param msglen: length of the message to expect from the peer
        :param maxfds: maximum number of file descriptors to expect from the peer
        :return: a tuple of (message, file descriptors)
        """