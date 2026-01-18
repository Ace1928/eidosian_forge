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
class FileResponse(Response):
    chunk_size = 64 * 1024

    def __init__(self, path: str | os.PathLike[str], status_code: int=200, headers: typing.Mapping[str, str] | None=None, media_type: str | None=None, background: BackgroundTask | None=None, filename: str | None=None, stat_result: os.stat_result | None=None, method: str | None=None, content_disposition_type: str='attachment') -> None:
        self.path = path
        self.status_code = status_code
        self.filename = filename
        if method is not None:
            warnings.warn("The 'method' parameter is not used, and it will be removed.", DeprecationWarning)
        if media_type is None:
            media_type = guess_type(filename or path)[0] or 'text/plain'
        self.media_type = media_type
        self.background = background
        self.init_headers(headers)
        if self.filename is not None:
            content_disposition_filename = quote(self.filename)
            if content_disposition_filename != self.filename:
                content_disposition = "{}; filename*=utf-8''{}".format(content_disposition_type, content_disposition_filename)
            else:
                content_disposition = '{}; filename="{}"'.format(content_disposition_type, self.filename)
            self.headers.setdefault('content-disposition', content_disposition)
        self.stat_result = stat_result
        if stat_result is not None:
            self.set_stat_headers(stat_result)

    def set_stat_headers(self, stat_result: os.stat_result) -> None:
        content_length = str(stat_result.st_size)
        last_modified = formatdate(stat_result.st_mtime, usegmt=True)
        etag_base = str(stat_result.st_mtime) + '-' + str(stat_result.st_size)
        etag = f'"{md5_hexdigest(etag_base.encode(), usedforsecurity=False)}"'
        self.headers.setdefault('content-length', content_length)
        self.headers.setdefault('last-modified', last_modified)
        self.headers.setdefault('etag', etag)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.stat_result is None:
            try:
                stat_result = await anyio.to_thread.run_sync(os.stat, self.path)
                self.set_stat_headers(stat_result)
            except FileNotFoundError:
                raise RuntimeError(f'File at path {self.path} does not exist.')
            else:
                mode = stat_result.st_mode
                if not stat.S_ISREG(mode):
                    raise RuntimeError(f'File at path {self.path} is not a file.')
        await send({'type': 'http.response.start', 'status': self.status_code, 'headers': self.raw_headers})
        if scope['method'].upper() == 'HEAD':
            await send({'type': 'http.response.body', 'body': b'', 'more_body': False})
        elif 'extensions' in scope and 'http.response.pathsend' in scope['extensions']:
            await send({'type': 'http.response.pathsend', 'path': str(self.path)})
        else:
            async with await anyio.open_file(self.path, mode='rb') as file:
                more_body = True
                while more_body:
                    chunk = await file.read(self.chunk_size)
                    more_body = len(chunk) == self.chunk_size
                    await send({'type': 'http.response.body', 'body': chunk, 'more_body': more_body})
        if self.background is not None:
            await self.background()