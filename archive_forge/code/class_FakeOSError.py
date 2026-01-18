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
class FakeOSError(OSError):
    pass