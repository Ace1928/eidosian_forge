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
class RedirectResponse(Response):

    def __init__(self, url: str | URL, status_code: int=307, headers: typing.Mapping[str, str] | None=None, background: BackgroundTask | None=None) -> None:
        super().__init__(content=b'', status_code=status_code, headers=headers, background=background)
        self.headers['location'] = quote(str(url), safe=":/%#?=@[]!$&'()*+,;")