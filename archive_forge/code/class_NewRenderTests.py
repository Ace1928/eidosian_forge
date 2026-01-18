import os
import zlib
from io import BytesIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import interfaces
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.task import Clock
from twisted.internet.testing import EventLoggingObserver, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python import failure, reflect
from twisted.python.compat import iterbytes
from twisted.python.filepath import FilePath
from twisted.trial import unittest
from twisted.web import error, http, iweb, resource, server
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Request, Site
from twisted.web.static import Data
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from ._util import assertIsFilesystemTemporary
class NewRenderTests(unittest.TestCase):
    """
    Tests for L{server.Request.render}.
    """

    def _getReq(self, resource=None):
        """
        Create a request object with a stub channel and install the
        passed resource at /newrender. If no resource is passed,
        create one.
        """
        d = DummyChannel()
        if resource is None:
            resource = NewRenderResource()
        d.site.resource.putChild(b'newrender', resource)
        d.transport.port = 81
        request = server.Request(d, 1)
        request.setHost(b'example.com', 81)
        request.gotLength(0)
        return request

    def testGoodMethods(self):
        req = self._getReq()
        req.requestReceived(b'GET', b'/newrender', b'HTTP/1.0')
        self.assertEqual(req.transport.written.getvalue().splitlines()[-1], b'hi hi')
        req = self._getReq()
        req.requestReceived(b'HEH', b'/newrender', b'HTTP/1.0')
        self.assertEqual(req.transport.written.getvalue().splitlines()[-1], b'ho ho')

    def testBadMethods(self):
        req = self._getReq()
        req.requestReceived(b'CONNECT', b'/newrender', b'HTTP/1.0')
        self.assertEqual(req.code, 501)
        req = self._getReq()
        req.requestReceived(b'hlalauguG', b'/newrender', b'HTTP/1.0')
        self.assertEqual(req.code, 501)

    def test_notAllowedMethod(self):
        """
        When trying to invoke a method not in the allowed method list, we get
        a response saying it is not allowed.
        """
        req = self._getReq()
        req.requestReceived(b'POST', b'/newrender', b'HTTP/1.0')
        self.assertEqual(req.code, 405)
        self.assertTrue(req.responseHeaders.hasHeader(b'allow'))
        raw_header = req.responseHeaders.getRawHeaders(b'allow')[0]
        allowed = sorted((h.strip() for h in raw_header.split(b',')))
        self.assertEqual([b'GET', b'HEAD', b'HEH'], allowed)

    def testImplicitHead(self):
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        req = self._getReq()
        req.requestReceived(b'HEAD', b'/newrender', b'HTTP/1.0')
        self.assertEqual(req.code, 200)
        self.assertEqual(-1, req.transport.written.getvalue().find(b'hi hi'))
        self.assertEquals(1, len(logObserver))
        event = logObserver[0]
        self.assertEquals(event['log_level'], LogLevel.info)

    def test_unsupportedHead(self):
        """
        HEAD requests against resource that only claim support for GET
        should not include a body in the response.
        """
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        resource = HeadlessResource()
        req = self._getReq(resource)
        req.requestReceived(b'HEAD', b'/newrender', b'HTTP/1.0')
        headers, body = req.transport.written.getvalue().split(b'\r\n\r\n')
        self.assertEqual(req.code, 200)
        self.assertEqual(body, b'')
        self.assertEquals(2, len(logObserver))

    def test_noBytesResult(self):
        """
        When implemented C{render} method does not return bytes an internal
        server error is returned.
        """

        class RiggedRepr:

            def __repr__(self) -> str:
                return 'my>repr'
        result = RiggedRepr()
        no_bytes_resource = resource.Resource()
        no_bytes_resource.render = lambda request: result
        request = self._getReq(no_bytes_resource)
        request.requestReceived(b'GET', b'/newrender', b'HTTP/1.0')
        headers, body = request.transport.written.getvalue().split(b'\r\n\r\n')
        self.assertEqual(request.code, 500)
        expected = ['', '<html>', '  <head><title>500 - Request did not return bytes</title></head>', '  <body>', '    <h1>Request did not return bytes</h1>', '    <p>Request: <pre>&lt;%s&gt;</pre><br />Resource: <pre>&lt;%s&gt;</pre><br />Value: <pre>my&gt;repr</pre></p>' % (reflect.safe_repr(request)[1:-1], reflect.safe_repr(no_bytes_resource)[1:-1]), '  </body>', '</html>', '']
        self.assertEqual('\n'.join(expected).encode('ascii'), body)