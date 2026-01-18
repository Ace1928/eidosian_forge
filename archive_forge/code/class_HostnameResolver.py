from __future__ import annotations
import socket
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar
import trio
class HostnameResolver(metaclass=ABCMeta):
    """If you have a custom hostname resolver, then implementing
    :class:`HostnameResolver` allows you to register this to be used by Trio.

    See :func:`trio.socket.set_custom_hostname_resolver`.

    """
    __slots__ = ()

    @abstractmethod
    async def getaddrinfo(self, host: bytes | None, port: bytes | str | int | None, family: int=0, type: int=0, proto: int=0, flags: int=0) -> list[tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int] | tuple[str, int, int, int]]]:
        """A custom implementation of :func:`~trio.socket.getaddrinfo`.

        Called by :func:`trio.socket.getaddrinfo`.

        If ``host`` is given as a numeric IP address, then
        :func:`~trio.socket.getaddrinfo` may handle the request itself rather
        than calling this method.

        Any required IDNA encoding is handled before calling this function;
        your implementation can assume that it will never see U-labels like
        ``"café.com"``, and only needs to handle A-labels like
        ``b"xn--caf-dma.com"``.

        """

    @abstractmethod
    async def getnameinfo(self, sockaddr: tuple[str, int] | tuple[str, int, int, int], flags: int) -> tuple[str, str]:
        """A custom implementation of :func:`~trio.socket.getnameinfo`.

        Called by :func:`trio.socket.getnameinfo`.

        """