from __future__ import annotations
import importlib.util
import os
import stat
import typing
from email.utils import parsedate
import anyio
import anyio.to_thread
from starlette._utils import get_route_path
from starlette.datastructures import URL, Headers
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, RedirectResponse, Response
from starlette.types import Receive, Scope, Send
class NotModifiedResponse(Response):
    NOT_MODIFIED_HEADERS = ('cache-control', 'content-location', 'date', 'etag', 'expires', 'vary')

    def __init__(self, headers: Headers):
        super().__init__(status_code=304, headers={name: value for name, value in headers.items() if name in self.NOT_MODIFIED_HEADERS})