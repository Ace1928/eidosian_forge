import ssl
import typing
from typing import Optional
from .._exceptions import ReadError
from .base import (
class MockSSLObject:

    def __init__(self, http2: bool):
        self._http2 = http2

    def selected_alpn_protocol(self) -> str:
        return 'h2' if self._http2 else 'http/1.1'