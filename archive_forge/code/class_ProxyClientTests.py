from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
class ProxyClientTests(TestCase):
    """
    Tests for L{ProxyClient}.
    """

    def _parseOutHeaders(self, content):
        """
        Parse the headers out of some web content.

        @param content: Bytes received from a web server.
        @return: A tuple of (requestLine, headers, body). C{headers} is a dict
            of headers, C{requestLine} is the first line (e.g. "POST /foo ...")
            and C{body} is whatever is left.
        """
        headers, body = content.split(b'\r\n\r\n')
        headers = headers.split(b'\r\n')
        requestLine = headers.pop(0)
        return (requestLine, dict((header.split(b': ') for header in headers)), body)

    def makeRequest(self, path):
        """
        Make a dummy request object for the URL path.

        @param path: A URL path, beginning with a slash.
        @return: A L{DummyRequest}.
        """
        return DummyRequest(path)

    def makeProxyClient(self, request, method=b'GET', headers=None, requestBody=b''):
        """
        Make a L{ProxyClient} object used for testing.

        @param request: The request to use.
        @param method: The HTTP method to use, GET by default.
        @param headers: The HTTP headers to use expressed as a dict. If not
            provided, defaults to {'accept': 'text/html'}.
        @param requestBody: The body of the request. Defaults to the empty
            string.
        @return: A L{ProxyClient}
        """
        if headers is None:
            headers = {b'accept': b'text/html'}
        path = b'/' + request.postpath
        return ProxyClient(method, path, b'HTTP/1.0', headers, requestBody, request)

    def connectProxy(self, proxyClient):
        """
        Connect a proxy client to a L{StringTransportWithDisconnection}.

        @param proxyClient: A L{ProxyClient}.
        @return: The L{StringTransportWithDisconnection}.
        """
        clientTransport = StringTransportWithDisconnection()
        clientTransport.protocol = proxyClient
        proxyClient.makeConnection(clientTransport)
        return clientTransport

    def assertForwardsHeaders(self, proxyClient, requestLine, headers):
        """
        Assert that C{proxyClient} sends C{headers} when it connects.

        @param proxyClient: A L{ProxyClient}.
        @param requestLine: The request line we expect to be sent.
        @param headers: A dict of headers we expect to be sent.
        @return: If the assertion is successful, return the request body as
            bytes.
        """
        self.connectProxy(proxyClient)
        requestContent = proxyClient.transport.value()
        receivedLine, receivedHeaders, body = self._parseOutHeaders(requestContent)
        self.assertEqual(receivedLine, requestLine)
        self.assertEqual(receivedHeaders, headers)
        return body

    def makeResponseBytes(self, code, message, headers, body):
        lines = [b'HTTP/1.0 ' + str(code).encode('ascii') + b' ' + message]
        for header, values in headers:
            for value in values:
                lines.append(header + b': ' + value)
        lines.extend([b'', body])
        return b'\r\n'.join(lines)

    def assertForwardsResponse(self, request, code, message, headers, body):
        """
        Assert that C{request} has forwarded a response from the server.

        @param request: A L{DummyRequest}.
        @param code: The expected HTTP response code.
        @param message: The expected HTTP message.
        @param headers: The expected HTTP headers.
        @param body: The expected response body.
        """
        self.assertEqual(request.responseCode, code)
        self.assertEqual(request.responseMessage, message)
        receivedHeaders = list(request.responseHeaders.getAllRawHeaders())
        receivedHeaders.sort()
        expectedHeaders = headers[:]
        expectedHeaders.sort()
        self.assertEqual(receivedHeaders, expectedHeaders)
        self.assertEqual(b''.join(request.written), body)

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

    def test_forward(self):
        """
        When connected to the server, L{ProxyClient} should send the saved
        request, with modifications of the headers, and then forward the result
        to the parent request.
        """
        return self._testDataForward(200, b'OK', [(b'Foo', [b'bar', b'baz'])], b'Some data\r\n')

    def test_postData(self):
        """
        Try to post content in the request, and check that the proxy client
        forward the body of the request.
        """
        return self._testDataForward(200, b'OK', [(b'Foo', [b'bar'])], b'Some data\r\n', b'POST', b'Some content')

    def test_statusWithMessage(self):
        """
        If the response contains a status with a message, it should be
        forwarded to the parent request with all the information.
        """
        return self._testDataForward(404, b'Not Found', [], b'')

    def test_contentLength(self):
        """
        If the response contains a I{Content-Length} header, the inbound
        request object should still only have C{finish} called on it once.
        """
        data = b'foo bar baz'
        return self._testDataForward(200, b'OK', [(b'Content-Length', [str(len(data)).encode('ascii')])], data)

    def test_losesConnection(self):
        """
        If the response contains a I{Content-Length} header, the outgoing
        connection is closed when all response body data has been received.
        """
        data = b'foo bar baz'
        return self._testDataForward(200, b'OK', [(b'Content-Length', [str(len(data)).encode('ascii')])], data, loseConnection=False)

    def test_headersCleanups(self):
        """
        The headers given at initialization should be modified:
        B{proxy-connection} should be removed if present, and B{connection}
        should be added.
        """
        client = ProxyClient(b'GET', b'/foo', b'HTTP/1.0', {b'accept': b'text/html', b'proxy-connection': b'foo'}, b'', None)
        self.assertEqual(client.headers, {b'accept': b'text/html', b'connection': b'close'})

    def test_keepaliveNotForwarded(self):
        """
        The proxy doesn't really know what to do with keepalive things from
        the remote server, so we stomp over any keepalive header we get from
        the client.
        """
        headers = {b'accept': b'text/html', b'keep-alive': b'300', b'connection': b'keep-alive'}
        expectedHeaders = headers.copy()
        expectedHeaders[b'connection'] = b'close'
        del expectedHeaders[b'keep-alive']
        client = ProxyClient(b'GET', b'/foo', b'HTTP/1.0', headers, b'', None)
        self.assertForwardsHeaders(client, b'GET /foo HTTP/1.0', expectedHeaders)

    def test_defaultHeadersOverridden(self):
        """
        L{server.Request} within the proxy sets certain response headers by
        default. When we get these headers back from the remote server, the
        defaults are overridden rather than simply appended.
        """
        request = self.makeRequest(b'foo')
        request.responseHeaders.setRawHeaders(b'server', [b'old-bar'])
        request.responseHeaders.setRawHeaders(b'date', [b'old-baz'])
        request.responseHeaders.setRawHeaders(b'content-type', [b'old/qux'])
        client = self.makeProxyClient(request, headers={b'accept': b'text/html'})
        self.connectProxy(client)
        headers = {b'Server': [b'bar'], b'Date': [b'2010-01-01'], b'Content-Type': [b'application/x-baz']}
        client.dataReceived(self.makeResponseBytes(200, b'OK', headers.items(), b''))
        self.assertForwardsResponse(request, 200, b'OK', list(headers.items()), b'')