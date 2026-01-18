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
def _writeToEmptyBodyContentLength(self, transport):
    """
        Write this request to the given transport using content-length to frame
        the (empty) body.

        @param transport: See L{writeTo}.
        @return: See L{writeTo}.
        """
    self._writeHeaders(transport, b'Content-Length: 0\r\n')
    return succeed(None)