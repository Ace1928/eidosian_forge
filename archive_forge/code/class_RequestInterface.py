from contextlib import contextmanager
from typing import Iterator, Optional, Union
from .._models import (
class RequestInterface:

    def request(self, method: Union[bytes, str], url: Union[URL, bytes, str], *, headers: HeaderTypes=None, content: Union[bytes, Iterator[bytes], None]=None, extensions: Optional[Extensions]=None) -> Response:
        method = enforce_bytes(method, name='method')
        url = enforce_url(url, name='url')
        headers = enforce_headers(headers, name='headers')
        headers = include_request_headers(headers, url=url, content=content)
        request = Request(method=method, url=url, headers=headers, content=content, extensions=extensions)
        response = self.handle_request(request)
        try:
            response.read()
        finally:
            response.close()
        return response

    @contextmanager
    def stream(self, method: Union[bytes, str], url: Union[URL, bytes, str], *, headers: HeaderTypes=None, content: Union[bytes, Iterator[bytes], None]=None, extensions: Optional[Extensions]=None) -> Iterator[Response]:
        method = enforce_bytes(method, name='method')
        url = enforce_url(url, name='url')
        headers = enforce_headers(headers, name='headers')
        headers = include_request_headers(headers, url=url, content=content)
        request = Request(method=method, url=url, headers=headers, content=content, extensions=extensions)
        response = self.handle_request(request)
        try:
            yield response
        finally:
            response.close()

    def handle_request(self, request: Request) -> Response:
        raise NotImplementedError()