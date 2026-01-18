from __future__ import annotations
import os
import socket as stdlib_socket
import ssl
import sys
import threading
from contextlib import asynccontextmanager, contextmanager, suppress
from functools import partial
from ssl import SSLContext
from typing import (
import pytest
from trio import StapledStream
from trio._tests.pytest_plugin import skip_if_optional_else_raise
from trio.abc import ReceiveStream, SendStream
from trio.testing import (
import trio
from .. import _core, socket as tsocket
from .._abc import Stream
from .._core import BrokenResourceError, ClosedResourceError
from .._core._tests.tutil import slow
from .._highlevel_generic import aclose_forcefully
from .._highlevel_open_tcp_stream import open_tcp_stream
from .._highlevel_socket import SocketListener, SocketStream
from .._ssl import NeedHandshakeError, SSLListener, SSLStream, _is_eof
from .._util import ConflictDetector
from ..testing import (
class NotAStream(Stream):

    async def wait_send_all_might_not_block(self) -> None:
        record.append('ok')

    async def aclose(self) -> None:
        raise AssertionError('Should not get called')

    async def receive_some(self, max_bytes: int | None=None) -> bytes | bytearray:
        raise AssertionError('Should not get called')

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        raise AssertionError('Should not get called')