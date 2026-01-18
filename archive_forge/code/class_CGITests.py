import json
import os
import sys
from io import BytesIO
from twisted.internet import address, error, interfaces, reactor
from twisted.internet.error import ConnectionLost
from twisted.python import failure, log, util
from twisted.trial import unittest
from twisted.web import client, http, http_headers, resource, server, twcgi
from twisted.web.http import INTERNAL_SERVER_ERROR, NOT_FOUND
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
import os, sys
import sys
import json
import os
import os
class CGITests(_StartServerAndTearDownMixin, unittest.TestCase):
    """
    Tests for L{twcgi.FilteredScript}.
    """
    if not interfaces.IReactorProcess.providedBy(reactor):
        skip = 'CGI tests require a functional reactor.spawnProcess()'

    def test_CGI(self):
        cgiFilename = self.writeCGI(DUMMY_CGI)
        portnum = self.startServer(cgiFilename)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        d = client.Agent(reactor).request(b'GET', url)
        d.addCallback(client.readBody)
        d.addCallback(self._testCGI_1)
        return d

    def _testCGI_1(self, res):
        self.assertEqual(res, b'cgi output' + os.linesep.encode('ascii'))

    def test_protectedServerAndDate(self):
        """
        If the CGI script emits a I{Server} or I{Date} header, these are
        ignored.
        """
        cgiFilename = self.writeCGI(SPECIAL_HEADER_CGI)
        portnum = self.startServer(cgiFilename)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        agent = client.Agent(reactor)
        d = agent.request(b'GET', url)
        d.addCallback(discardBody)

        def checkResponse(response):
            self.assertNotIn('monkeys', response.headers.getRawHeaders('server'))
            self.assertNotIn('last year', response.headers.getRawHeaders('date'))
        d.addCallback(checkResponse)
        return d

    def test_noDuplicateContentTypeHeaders(self):
        """
        If the CGI script emits a I{content-type} header, make sure that the
        server doesn't add an additional (duplicate) one, as per ticket 4786.
        """
        cgiFilename = self.writeCGI(NO_DUPLICATE_CONTENT_TYPE_HEADER_CGI)
        portnum = self.startServer(cgiFilename)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        agent = client.Agent(reactor)
        d = agent.request(b'GET', url)
        d.addCallback(discardBody)

        def checkResponse(response):
            self.assertEqual(response.headers.getRawHeaders('content-type'), ['text/cgi-duplicate-test'])
            return response
        d.addCallback(checkResponse)
        return d

    def test_noProxyPassthrough(self):
        """
        The CGI script is never called with the Proxy header passed through.
        """
        cgiFilename = self.writeCGI(HEADER_OUTPUT_CGI)
        portnum = self.startServer(cgiFilename)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        agent = client.Agent(reactor)
        headers = http_headers.Headers({b'Proxy': [b'foo'], b'X-Innocent-Header': [b'bar']})
        d = agent.request(b'GET', url, headers=headers)

        def checkResponse(response):
            headers = json.loads(response.decode('ascii'))
            self.assertEqual(set(headers.keys()), {'HTTP_HOST', 'HTTP_CONNECTION', 'HTTP_X_INNOCENT_HEADER'})
        d.addCallback(client.readBody)
        d.addCallback(checkResponse)
        return d

    def test_duplicateHeaderCGI(self):
        """
        If a CGI script emits two instances of the same header, both are sent
        in the response.
        """
        cgiFilename = self.writeCGI(DUAL_HEADER_CGI)
        portnum = self.startServer(cgiFilename)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        agent = client.Agent(reactor)
        d = agent.request(b'GET', url)
        d.addCallback(discardBody)

        def checkResponse(response):
            self.assertEqual(response.headers.getRawHeaders('header'), ['spam', 'eggs'])
        d.addCallback(checkResponse)
        return d

    def test_malformedHeaderCGI(self):
        """
        Check for the error message in the duplicated header
        """
        cgiFilename = self.writeCGI(BROKEN_HEADER_CGI)
        portnum = self.startServer(cgiFilename)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        agent = client.Agent(reactor)
        d = agent.request(b'GET', url)
        d.addCallback(discardBody)
        loggedMessages = []

        def addMessage(eventDict):
            loggedMessages.append(log.textFromEventDict(eventDict))
        log.addObserver(addMessage)
        self.addCleanup(log.removeObserver, addMessage)

        def checkResponse(ignored):
            self.assertIn('ignoring malformed CGI header: ' + repr(b'XYZ'), loggedMessages)
        d.addCallback(checkResponse)
        return d

    def test_ReadEmptyInput(self):
        cgiFilename = os.path.abspath(self.mktemp())
        with open(cgiFilename, 'wt') as cgiFile:
            cgiFile.write(READINPUT_CGI)
        portnum = self.startServer(cgiFilename)
        agent = client.Agent(reactor)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        d = agent.request(b'GET', url)
        d.addCallback(client.readBody)
        d.addCallback(self._test_ReadEmptyInput_1)
        return d
    test_ReadEmptyInput.timeout = 5

    def _test_ReadEmptyInput_1(self, res):
        expected = f'readinput ok{os.linesep}'
        expected = expected.encode('ascii')
        self.assertEqual(res, expected)

    def test_ReadInput(self):
        cgiFilename = os.path.abspath(self.mktemp())
        with open(cgiFilename, 'wt') as cgiFile:
            cgiFile.write(READINPUT_CGI)
        portnum = self.startServer(cgiFilename)
        agent = client.Agent(reactor)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        d = agent.request(uri=url, method=b'POST', bodyProducer=client.FileBodyProducer(BytesIO(b'Here is your stdin')))
        d.addCallback(client.readBody)
        d.addCallback(self._test_ReadInput_1)
        return d
    test_ReadInput.timeout = 5

    def _test_ReadInput_1(self, res):
        expected = f'readinput ok{os.linesep}'
        expected = expected.encode('ascii')
        self.assertEqual(res, expected)

    def test_ReadAllInput(self):
        cgiFilename = os.path.abspath(self.mktemp())
        with open(cgiFilename, 'wt') as cgiFile:
            cgiFile.write(READALLINPUT_CGI)
        portnum = self.startServer(cgiFilename)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        d = client.Agent(reactor).request(uri=url, method=b'POST', bodyProducer=client.FileBodyProducer(BytesIO(b'Here is your stdin')))
        d.addCallback(client.readBody)
        d.addCallback(self._test_ReadAllInput_1)
        return d
    test_ReadAllInput.timeout = 5

    def _test_ReadAllInput_1(self, res):
        expected = f'readallinput ok{os.linesep}'
        expected = expected.encode('ascii')
        self.assertEqual(res, expected)

    def test_useReactorArgument(self):
        """
        L{twcgi.FilteredScript.runProcess} uses the reactor passed as an
        argument to the constructor.
        """

        class FakeReactor:
            """
            A fake reactor recording whether spawnProcess is called.
            """
            called = False

            def spawnProcess(self, *args, **kwargs):
                """
                Set the C{called} flag to C{True} if C{spawnProcess} is called.

                @param args: Positional arguments.
                @param kwargs: Keyword arguments.
                """
                self.called = True
        fakeReactor = FakeReactor()
        request = DummyRequest(['a', 'b'])
        request.client = address.IPv4Address('TCP', '127.0.0.1', 12345)
        resource = twcgi.FilteredScript('dummy-file', reactor=fakeReactor)
        _render(resource, request)
        self.assertTrue(fakeReactor.called)