from __future__ import annotations
import datetime
import email.message
import json as jsonlib
import typing
import urllib.request
from collections.abc import Mapping
from http.cookiejar import Cookie, CookieJar
from ._content import ByteStream, UnattachedStream, encode_request, encode_response
from ._decoders import (
from ._exceptions import (
from ._multipart import get_multipart_boundary_from_content_type
from ._status_codes import codes
from ._types import (
from ._urls import URL
from ._utils import (
class _CookieCompatRequest(urllib.request.Request):
    """
        Wraps a `Request` instance up in a compatibility interface suitable
        for use with `CookieJar` operations.
        """

    def __init__(self, request: Request) -> None:
        super().__init__(url=str(request.url), headers=dict(request.headers), method=request.method)
        self.request = request

    def add_unredirected_header(self, key: str, value: str) -> None:
        super().add_unredirected_header(key, value)
        self.request.headers[key] = value