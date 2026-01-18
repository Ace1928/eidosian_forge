from __future__ import annotations
import socket
import sys
from socket import AddressFamily, SocketKind
from typing import TYPE_CHECKING, Any, Sequence
import attrs
import pytest
import trio
from trio._highlevel_open_tcp_stream import (
from trio.socket import AF_INET, AF_INET6, IPPROTO_TCP, SOCK_STREAM, SocketType
from trio.testing import Matcher, RaisesGroup
def _ip_to_gai_entry(self, ip: str) -> tuple[AddressFamily, SocketKind, int, str, tuple[str, int, int, int] | tuple[str, int]]:
    sockaddr: tuple[str, int] | tuple[str, int, int, int]
    if ':' in ip:
        family = trio.socket.AF_INET6
        sockaddr = (ip, self.port, 0, 0)
    else:
        family = trio.socket.AF_INET
        sockaddr = (ip, self.port)
    return (family, SOCK_STREAM, IPPROTO_TCP, '', sockaddr)