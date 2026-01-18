from __future__ import annotations
import errno
import inspect
import os
import socket as stdlib_socket
import sys
import tempfile
from socket import AddressFamily, SocketKind
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union
import attrs
import pytest
from .. import _core, socket as tsocket
from .._core._tests.tutil import binds_ipv6, creates_ipv6
from .._socket import _NUMERIC_ONLY, SocketType, _SocketType, _try_sync
from ..testing import assert_checkpoints, wait_all_tasks_blocked
class MySocket(stdlib_socket.socket):
    pass