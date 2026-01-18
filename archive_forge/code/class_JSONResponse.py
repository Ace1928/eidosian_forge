from __future__ import annotations
import http.cookies
import json
import os
import stat
import typing
import warnings
from datetime import datetime
from email.utils import format_datetime, formatdate
from functools import partial
from mimetypes import guess_type
from urllib.parse import quote
import anyio
import anyio.to_thread
from starlette._compat import md5_hexdigest
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import URL, MutableHeaders
from starlette.types import Receive, Scope, Send
class JSONResponse(Response):
    media_type = 'application/json'

    def __init__(self, content: typing.Any, status_code: int=200, headers: typing.Mapping[str, str] | None=None, media_type: str | None=None, background: BackgroundTask | None=None) -> None:
        super().__init__(content, status_code, headers, media_type, background)

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False, indent=None, separators=(',', ':')).encode('utf-8')