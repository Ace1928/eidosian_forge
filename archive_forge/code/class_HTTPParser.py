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
class HTTPParser(LineReceiver):
    """
    L{HTTPParser} handles the parsing side of HTTP processing. With a suitable
    subclass, it can parse either the client side or the server side of the
    connection.

    @ivar headers: All of the non-connection control message headers yet
        received.

    @ivar state: State indicator for the response parsing state machine.  One
        of C{STATUS}, C{HEADER}, C{BODY}, C{DONE}.

    @ivar _partialHeader: L{None} or a C{list} of the lines of a multiline
        header while that header is being received.
    """
    delimiter = b'\n'
    CONNECTION_CONTROL_HEADERS = {b'content-length', b'connection', b'keep-alive', b'te', b'trailers', b'transfer-encoding', b'upgrade', b'proxy-connection'}

    def connectionMade(self):
        self.headers = Headers()
        self.connHeaders = Headers()
        self.state = STATUS
        self._partialHeader = None

    def switchToBodyMode(self, decoder):
        """
        Switch to body parsing mode - interpret any more bytes delivered as
        part of the message body and deliver them to the given decoder.
        """
        if self.state == BODY:
            raise RuntimeError('already in body mode')
        self.bodyDecoder = decoder
        self.state = BODY
        self.setRawMode()

    def lineReceived(self, line):
        """
        Handle one line from a response.
        """
        if line[-1:] == b'\r':
            line = line[:-1]
        if self.state == STATUS:
            self.statusReceived(line)
            self.state = HEADER
        elif self.state == HEADER:
            if not line or line[0] not in b' \t':
                if self._partialHeader is not None:
                    header = b''.join(self._partialHeader)
                    name, value = header.split(b':', 1)
                    value = value.strip()
                    self.headerReceived(name, value)
                if not line:
                    self.allHeadersReceived()
                else:
                    self._partialHeader = [line]
            else:
                self._partialHeader.append(line)

    def rawDataReceived(self, data):
        """
        Pass data from the message body to the body decoder object.
        """
        self.bodyDecoder.dataReceived(data)

    def isConnectionControlHeader(self, name):
        """
        Return C{True} if the given lower-cased name is the name of a
        connection control header (rather than an entity header).

        According to RFC 2616, section 14.10, the tokens in the Connection
        header are probably relevant here.  However, I am not sure what the
        practical consequences of either implementing or ignoring that are.
        So I leave it unimplemented for the time being.
        """
        return name in self.CONNECTION_CONTROL_HEADERS

    def statusReceived(self, status):
        """
        Callback invoked whenever the first line of a new message is received.
        Override this.

        @param status: The first line of an HTTP request or response message
            without trailing I{CR LF}.
        @type status: C{bytes}
        """

    def headerReceived(self, name, value):
        """
        Store the given header in C{self.headers}.
        """
        name = name.lower()
        if self.isConnectionControlHeader(name):
            headers = self.connHeaders
        else:
            headers = self.headers
        headers.addRawHeader(name, value)

    def allHeadersReceived(self):
        """
        Callback invoked after the last header is passed to C{headerReceived}.
        Override this to change to the C{BODY} or C{DONE} state.
        """
        self.switchToBodyMode(None)