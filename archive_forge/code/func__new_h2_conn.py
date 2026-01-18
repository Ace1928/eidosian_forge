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
def _new_h2_conn(self) -> _LockedObject[h2.connection.H2Connection]:
    config = h2.config.H2Configuration(client_side=True)
    return _LockedObject(h2.connection.H2Connection(config=config))