from __future__ import annotations
import errno
import socket as stdlib_socket
import sys
from socket import AddressFamily, SocketKind
from typing import TYPE_CHECKING, Any, Sequence, overload
import attrs
import pytest
import trio
from trio import (
from trio.abc import HostnameResolver, SendStream, SocketFactory
from trio.testing import open_stream_to_socket_listener
from .. import socket as tsocket
from .._core._tests.tutil import binds_ipv6
@attrs.define(slots=False)
class FakeSocketFactory(SocketFactory):
    poison_after: int
    sockets: list[tsocket.SocketType] = attrs.Factory(list)
    raise_on_family: dict[AddressFamily, int] = attrs.Factory(dict)

    def socket(self, family: AddressFamily | int | None=None, type_: SocketKind | int | None=None, proto: int=0) -> tsocket.SocketType:
        assert family is not None
        assert type_ is not None
        if isinstance(family, int) and (not isinstance(family, AddressFamily)):
            family = AddressFamily(family)
        if family in self.raise_on_family:
            raise OSError(self.raise_on_family[family], 'nope')
        sock = FakeSocket(family, type_, proto)
        self.poison_after -= 1
        if self.poison_after == 0:
            sock.poison_listen = True
        self.sockets.append(sock)
        return sock