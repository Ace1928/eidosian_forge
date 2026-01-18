from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def _testDataForward(self, code, message, headers, body, method=b'GET', requestBody=b'', loseConnection=True):
    """
        Build a fake proxy connection, and send C{data} over it, checking that
        it's forwarded to the originating request.
        """
    request = self.makeRequest(b'foo')
    client = self.makeProxyClient(request, method, {b'accept': b'text/html'}, requestBody)
    receivedBody = self.assertForwardsHeaders(client, method + b' /foo HTTP/1.0', {b'connection': b'close', b'accept': b'text/html'})
    self.assertEqual(receivedBody, requestBody)
    client.dataReceived(self.makeResponseBytes(code, message, headers, body))
    self.assertForwardsResponse(request, code, message, headers, body)
    if loseConnection:
        client.transport.loseConnection()
    self.assertFalse(client.transport.connected)
    self.assertEqual(request.finished, 1)