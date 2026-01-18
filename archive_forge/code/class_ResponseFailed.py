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
class ResponseFailed(_WrapperException):
    """
    L{ResponseFailed} indicates that all of the response to a request was not
    received for some reason.

    @ivar reasons: A C{list} of one or more L{Failure} instances giving the
        reasons the response was considered to have failed.

    @ivar response: If specified, the L{Response} received from the server (and
        in particular the status code and the headers).
    """

    def __init__(self, reasons, response=None):
        _WrapperException.__init__(self, reasons)
        self.response = response