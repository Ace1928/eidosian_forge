from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Union
from .._models import (
class AsyncRequestInterface:

    async def request(self, method: Union[bytes, str], url: Union[URL, bytes, str], *, headers: HeaderTypes=None, content: Union[bytes, AsyncIterator[bytes], None]=None, extensions: Optional[Extensions]=None) -> Response:
        method = enforce_bytes(method, name='method')
        url = enforce_url(url, name='url')
        headers = enforce_headers(headers, name='headers')
        headers = include_request_headers(headers, url=url, content=content)
        request = Request(method=method, url=url, headers=headers, content=content, extensions=extensions)
        response = await self.handle_async_request(request)
        try:
            await response.aread()
        finally:
            await response.aclose()
        return response

    @asynccontextmanager
    async def stream(self, method: Union[bytes, str], url: Union[URL, bytes, str], *, headers: HeaderTypes=None, content: Union[bytes, AsyncIterator[bytes], None]=None, extensions: Optional[Extensions]=None) -> AsyncIterator[Response]:
        method = enforce_bytes(method, name='method')
        url = enforce_url(url, name='url')
        headers = enforce_headers(headers, name='headers')
        headers = include_request_headers(headers, url=url, content=content)
        request = Request(method=method, url=url, headers=headers, content=content, extensions=extensions)
        response = await self.handle_async_request(request)
        try:
            yield response
        finally:
            await response.aclose()

    async def handle_async_request(self, request: Request) -> Response:
        raise NotImplementedError()