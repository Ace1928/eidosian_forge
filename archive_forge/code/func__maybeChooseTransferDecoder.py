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
def _maybeChooseTransferDecoder(self, header, data):
    """
        If the provided header is C{content-length} or
        C{transfer-encoding}, choose the appropriate decoder if any.

        Returns L{True} if the request can proceed and L{False} if not.
        """

    def fail():
        self._respondToBadRequestAndDisconnect()
        self.length = None
        return False
    if header == b'content-length':
        if not data.isdigit():
            return fail()
        try:
            length = int(data)
        except ValueError:
            return fail()
        newTransferDecoder = _IdentityTransferDecoder(length, self.requests[-1].handleContentChunk, self._finishRequestBody)
    elif header == b'transfer-encoding':
        if data.lower() == b'chunked':
            length = None
            newTransferDecoder = _ChunkedTransferDecoder(self.requests[-1].handleContentChunk, self._finishRequestBody)
        elif data.lower() == b'identity':
            return True
        else:
            return fail()
    else:
        return True
    if self._transferDecoder is not None:
        return fail()
    else:
        self.length = length
        self._transferDecoder = newTransferDecoder
        return True