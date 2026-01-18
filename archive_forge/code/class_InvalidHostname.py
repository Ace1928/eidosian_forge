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
class InvalidHostname(H2Error):

    def __init__(self, request: Request, expected_hostname: str, expected_netloc: str) -> None:
        self.request = request
        self.expected_hostname = expected_hostname
        self.expected_netloc = expected_netloc

    def __str__(self) -> str:
        return f'InvalidHostname: Expected {self.expected_hostname} or {self.expected_netloc} in {self.request}'