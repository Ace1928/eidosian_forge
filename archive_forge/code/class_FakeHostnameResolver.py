from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING, Any, NoReturn
import attrs
import pytest
import trio
import trio.testing
from trio.socket import AF_INET, IPPROTO_TCP, SOCK_STREAM
from .._highlevel_ssl_helpers import (
from .test_ssl import SERVER_CTX, client_ctx  # noqa: F401
@attrs.define(slots=False)
class FakeHostnameResolver(trio.abc.HostnameResolver):
    sockaddr: tuple[str, int] | tuple[str, int, int, int]

    async def getaddrinfo(self, host: bytes | None, port: bytes | str | int | None, family: int=0, type: int=0, proto: int=0, flags: int=0) -> list[tuple[AddressFamily, SocketKind, int, str, tuple[str, int] | tuple[str, int, int, int]]]:
        return [(AF_INET, SOCK_STREAM, IPPROTO_TCP, '', self.sockaddr)]

    async def getnameinfo(self, *args: Any) -> NoReturn:
        raise NotImplementedError