from __future__ import annotations
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol, TypeVar
import trio
from trio.socket import SOCK_STREAM, socket
class Closable(Protocol):

    def close(self) -> None:
        ...