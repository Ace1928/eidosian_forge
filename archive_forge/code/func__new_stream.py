import ipaddress
import itertools
import logging
from collections import deque
from ipaddress import IPv4Address, IPv6Address
from typing import Dict, List, Optional, Union
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.errors import ErrorCodes
from h2.events import (
from h2.exceptions import FrameTooLargeError, H2Error
from twisted.internet.defer import Deferred
from twisted.internet.error import TimeoutError
from twisted.internet.interfaces import IHandshakeListener, IProtocolNegotiationFactory
from twisted.internet.protocol import Factory, Protocol, connectionDone
from twisted.internet.ssl import Certificate
from twisted.protocols.policies import TimeoutMixin
from twisted.python.failure import Failure
from twisted.web.client import URI
from zope.interface import implementer
from scrapy.core.http2.stream import Stream, StreamCloseReason
from scrapy.http import Request
from scrapy.settings import Settings
from scrapy.spiders import Spider
def _new_stream(self, request: Request, spider: Spider) -> Stream:
    """Instantiates a new Stream object"""
    stream = Stream(stream_id=next(self._stream_id_generator), request=request, protocol=self, download_maxsize=getattr(spider, 'download_maxsize', self.metadata['default_download_maxsize']), download_warnsize=getattr(spider, 'download_warnsize', self.metadata['default_download_warnsize']))
    self.streams[stream.stream_id] = stream
    return stream