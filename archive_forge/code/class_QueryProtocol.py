import base64
import xmlrpc.client as xmlrpclib
from urllib.parse import urlparse
from xmlrpc.client import Binary, Boolean, DateTime, Fault
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure, reflect
from twisted.python.compat import nativeString
from twisted.web import http, resource, server
class QueryProtocol(http.HTTPClient):

    def connectionMade(self):
        self._response = None
        self.sendCommand(b'POST', self.factory.path)
        self.sendHeader(b'User-Agent', b'Twisted/XMLRPClib')
        self.sendHeader(b'Host', self.factory.host)
        self.sendHeader(b'Content-type', b'text/xml; charset=utf-8')
        payload = self.factory.payload
        self.sendHeader(b'Content-length', b'%d' % (len(payload),))
        if self.factory.user:
            auth = b':'.join([self.factory.user, self.factory.password])
            authHeader = b''.join([b'Basic ', base64.b64encode(auth)])
            self.sendHeader(b'Authorization', authHeader)
        self.endHeaders()
        self.transport.write(payload)

    def handleStatus(self, version, status, message):
        if status != b'200':
            self.factory.badStatus(status, message)

    def handleResponse(self, contents):
        """
        Handle the XML-RPC response received from the server.

        Specifically, disconnect from the server and store the XML-RPC
        response so that it can be properly handled when the disconnect is
        finished.
        """
        self.transport.loseConnection()
        self._response = contents

    def connectionLost(self, reason):
        """
        The connection to the server has been lost.

        If we have a full response from the server, then parse it and fired a
        Deferred with the return value or C{Fault} that the server gave us.
        """
        if not reason.check(error.ConnectionDone, error.ConnectionLost):
            self.factory.clientConnectionLost(None, reason)
        http.HTTPClient.connectionLost(self, reason)
        if self._response is not None:
            response, self._response = (self._response, None)
            self.factory.parseResponse(response)