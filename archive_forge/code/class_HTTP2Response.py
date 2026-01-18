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
class HTTP2Response(BaseHTTPResponse):

    def __init__(self, status: int, headers: HTTPHeaderDict, request_url: str, data: bytes, decode_content: bool=False) -> None:
        super().__init__(status=status, headers=headers, version=20, reason=None, decode_content=decode_content, request_url=request_url)
        self._data = data
        self.length_remaining = 0

    @property
    def data(self) -> bytes:
        return self._data

    def get_redirect_location(self) -> None:
        return None

    def close(self) -> None:
        pass