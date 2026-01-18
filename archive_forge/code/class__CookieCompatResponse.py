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
class _CookieCompatResponse:
    """
        Wraps a `Request` instance up in a compatibility interface suitable
        for use with `CookieJar` operations.
        """

    def __init__(self, response: Response) -> None:
        self.response = response

    def info(self) -> email.message.Message:
        info = email.message.Message()
        for key, value in self.response.headers.multi_items():
            info[key] = value
        return info