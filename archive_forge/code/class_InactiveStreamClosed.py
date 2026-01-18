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
class InactiveStreamClosed(ConnectionClosed):
    """Connection was closed without sending request headers
    of the stream. This happens when a stream is waiting for other
    streams to close and connection is lost."""

    def __init__(self, request: Request) -> None:
        self.request = request

    def __str__(self) -> str:
        return f'InactiveStreamClosed: Connection was closed without sending the request {self.request!r}'