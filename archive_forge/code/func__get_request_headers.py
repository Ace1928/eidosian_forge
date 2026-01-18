import logging
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from h2.errors import ErrorCodes
from h2.exceptions import H2Error, ProtocolError, StreamClosedError
from hpack import HeaderTuple
from twisted.internet.defer import CancelledError, Deferred
from twisted.internet.error import ConnectionClosed
from twisted.python.failure import Failure
from twisted.web.client import ResponseFailed
from scrapy.http import Request
from scrapy.http.headers import Headers
from scrapy.responsetypes import responsetypes
def _get_request_headers(self) -> List[Tuple[str, str]]:
    url = urlparse(self._request.url)
    path = url.path
    if url.query:
        path += '?' + url.query
    if not path:
        path = '*' if self._request.method == 'OPTIONS' else '/'
    headers = [(':method', self._request.method), (':authority', url.netloc)]
    if self._request.method != 'CONNECT':
        headers += [(':scheme', self._protocol.metadata['uri'].scheme), (':path', path)]
    content_length = str(len(self._request.body))
    headers.append(('Content-Length', content_length))
    content_length_name = self._request.headers.normkey(b'Content-Length')
    for name, values in self._request.headers.items():
        for value in values:
            value = str(value, 'utf-8')
            if name == content_length_name:
                if value != content_length:
                    logger.warning('Ignoring bad Content-Length header %r of request %r, sending %r instead', value, self._request, content_length)
                continue
            headers.append((str(name, 'utf-8'), value))
    return headers