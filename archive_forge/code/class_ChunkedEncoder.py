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
@implementer(IConsumer)
class ChunkedEncoder:
    """
    Helper object which exposes L{IConsumer} on top of L{HTTP11ClientProtocol}
    for streaming request bodies to the server.
    """

    def __init__(self, transport):
        self.transport = transport

    def _allowNoMoreWrites(self):
        """
        Indicate that no additional writes are allowed.  Attempts to write
        after calling this method will be met with an exception.
        """
        self.transport = None

    def registerProducer(self, producer, streaming):
        """
        Register the given producer with C{self.transport}.
        """
        self.transport.registerProducer(producer, streaming)

    def write(self, data):
        """
        Write the given request body bytes to the transport using chunked
        encoding.

        @type data: C{bytes}
        """
        if self.transport is None:
            raise ExcessWrite()
        self.transport.writeSequence((networkString('%x\r\n' % len(data)), data, b'\r\n'))

    def unregisterProducer(self):
        """
        Indicate that the request body is complete and finish the request.
        """
        self.write(b'')
        self.transport.unregisterProducer()
        self._allowNoMoreWrites()