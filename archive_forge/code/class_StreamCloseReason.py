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
class StreamCloseReason(Enum):
    ENDED = 1
    RESET = 2
    CONNECTION_LOST = 3
    MAXSIZE_EXCEEDED = 4
    CANCELLED = 5
    INACTIVE = 6
    INVALID_HOSTNAME = 7