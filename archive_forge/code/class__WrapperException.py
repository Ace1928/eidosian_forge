import re
from zope.interface import implementer
from twisted.internet.defer import (
from twisted.internet.error import ConnectionDone
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.reflect import fullyQualifiedName
from twisted.web.http import (
from twisted.web.http_headers import Headers
from twisted.web.iweb import UNKNOWN_LENGTH, IClientRequest, IResponse
class _WrapperException(Exception):
    """
    L{_WrapperException} is the base exception type for exceptions which
    include one or more other exceptions as the low-level causes.

    @ivar reasons: A L{list} of one or more L{Failure} instances encountered
        during an HTTP request.  See subclass documentation for more details.
    """

    def __init__(self, reasons):
        Exception.__init__(self, reasons)
        self.reasons = reasons