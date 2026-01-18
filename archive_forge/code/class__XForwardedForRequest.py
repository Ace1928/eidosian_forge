from __future__ import annotations
import base64
import binascii
import calendar
import math
import os
import re
import tempfile
import time
import warnings
from email import message_from_bytes
from email.message import EmailMessage
from io import BytesIO
from typing import AnyStr, Callable, Dict, List, Optional, Tuple
from urllib.parse import (
from zope.interface import Attribute, Interface, implementer, provider
from incremental import Version
from twisted.internet import address, interfaces, protocol
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IProtocol
from twisted.logger import Logger
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString, networkString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import deprecated
from twisted.python.failure import Failure
from twisted.web._responses import (
from twisted.web.http_headers import Headers, _sanitizeLinearWhitespace
from twisted.web.iweb import IAccessLogFormatter, INonQueuedRequestFactory, IRequest
class _XForwardedForRequest(proxyForInterface(IRequest, '_request')):
    """
    Add a layer on top of another request that only uses the value of an
    X-Forwarded-For header as the result of C{getClientAddress}.
    """

    def getClientAddress(self):
        """
        The client address (the first address) in the value of the
        I{X-Forwarded-For header}.  If the header is not present, the IP is
        considered to be C{b"-"}.

        @return: L{_XForwardedForAddress} which wraps the client address as
            expected by L{combinedLogFormatter}.
        """
        host = self._request.requestHeaders.getRawHeaders(b'x-forwarded-for', [b'-'])[0].split(b',')[0].strip()
        return _XForwardedForAddress(host)

    @property
    def clientproto(self):
        """
        @return: The protocol version in the request.
        @rtype: L{bytes}
        """
        return self._request.clientproto

    @property
    def code(self):
        """
        @return: The response code for the request.
        @rtype: L{int}
        """
        return self._request.code

    @property
    def sentLength(self):
        """
        @return: The number of bytes sent in the response body.
        @rtype: L{int}
        """
        return self._request.sentLength