from __future__ import annotations
import threading
import types
import typing
import h2.config  # type: ignore[import-untyped]
import h2.connection  # type: ignore[import-untyped]
import h2.events  # type: ignore[import-untyped]
import urllib3.connection
import urllib3.util.ssl_
from urllib3.response import BaseHTTPResponse
from ._collections import HTTPHeaderDict
from .connection import HTTPSConnection
from .connectionpool import HTTPSConnectionPool
class _LockedObject(typing.Generic[T]):
    """
    A wrapper class that hides a specific object behind a lock.

    The goal here is to provide a simple way to protect access to an object
    that cannot safely be simultaneously accessed from multiple threads. The
    intended use of this class is simple: take hold of it with a context
    manager, which returns the protected object.
    """

    def __init__(self, obj: T):
        self.lock = threading.RLock()
        self._obj = obj

    def __enter__(self) -> T:
        self.lock.acquire()
        return self._obj

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None:
        self.lock.release()