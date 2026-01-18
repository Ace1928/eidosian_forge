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
class StreamingResponse(Response):
    body_iterator: AsyncContentStream

    def __init__(self, content: ContentStream, status_code: int=200, headers: typing.Mapping[str, str] | None=None, media_type: str | None=None, background: BackgroundTask | None=None) -> None:
        if isinstance(content, typing.AsyncIterable):
            self.body_iterator = content
        else:
            self.body_iterator = iterate_in_threadpool(content)
        self.status_code = status_code
        self.media_type = self.media_type if media_type is None else media_type
        self.background = background
        self.init_headers(headers)

    async def listen_for_disconnect(self, receive: Receive) -> None:
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                break

    async def stream_response(self, send: Send) -> None:
        await send({'type': 'http.response.start', 'status': self.status_code, 'headers': self.raw_headers})
        async for chunk in self.body_iterator:
            if not isinstance(chunk, bytes):
                chunk = chunk.encode(self.charset)
            await send({'type': 'http.response.body', 'body': chunk, 'more_body': True})
        await send({'type': 'http.response.body', 'body': b'', 'more_body': False})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async with anyio.create_task_group() as task_group:

            async def wrap(func: typing.Callable[[], typing.Awaitable[None]]) -> None:
                await func()
                task_group.cancel_scope.cancel()
            task_group.start_soon(wrap, partial(self.stream_response, send))
            await wrap(partial(self.listen_for_disconnect, receive))
        if self.background is not None:
            await self.background()