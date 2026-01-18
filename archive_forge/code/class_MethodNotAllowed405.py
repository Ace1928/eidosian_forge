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
class MethodNotAllowed405(H2Error):

    def __init__(self, remote_ip_address: Optional[Union[IPv4Address, IPv6Address]]) -> None:
        self.remote_ip_address = remote_ip_address

    def __str__(self) -> str:
        return f"Received 'HTTP/2.0 405 Method Not Allowed' from {self.remote_ip_address!r}"