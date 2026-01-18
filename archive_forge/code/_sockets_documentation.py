from __future__ import annotations
import errno
import os
import socket
import ssl
import stat
import sys
from collections.abc import Awaitable
from ipaddress import IPv6Address, ip_address
from os import PathLike, chmod
from socket import AddressFamily, SocketKind
from typing import Any, Literal, cast, overload
from .. import to_thread
from ..abc import (
from ..streams.stapled import MultiListener
from ..streams.tls import TLSStream
from ._eventloop import get_async_backend
from ._resources import aclose_forcefully
from ._synchronization import Event
from ._tasks import create_task_group, move_on_after

    Create a UNIX local socket object, deleting the socket at the given path if it
    exists.

    Not available on Windows.

    :param path: path of the socket
    :param mode: permissions to set on the socket
    :param socktype: socket.SOCK_STREAM or socket.SOCK_DGRAM

    