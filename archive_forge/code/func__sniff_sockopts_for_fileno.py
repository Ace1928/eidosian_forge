from __future__ import annotations
import os
import select
import socket as _stdlib_socket
import sys
from operator import index
from socket import AddressFamily, SocketKind
from typing import (
import idna as _idna
import trio
from trio._util import wraps as _wraps
from . import _core
def _sniff_sockopts_for_fileno(family: AddressFamily | int, type_: SocketKind | int, proto: int, fileno: int | None) -> tuple[AddressFamily | int, SocketKind | int, int]:
    """Correct SOCKOPTS for given fileno, falling back to provided values."""
    if sys.platform != 'linux':
        return (family, type_, proto)
    from socket import SO_DOMAIN, SO_PROTOCOL, SO_TYPE, SOL_SOCKET
    sockobj = _stdlib_socket.socket(family, type_, proto, fileno=fileno)
    try:
        family = sockobj.getsockopt(SOL_SOCKET, SO_DOMAIN)
        proto = sockobj.getsockopt(SOL_SOCKET, SO_PROTOCOL)
        type_ = sockobj.getsockopt(SOL_SOCKET, SO_TYPE)
    finally:
        sockobj.detach()
    return (family, type_, proto)