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
class AllowedMethodsTests(unittest.TestCase):
    """
    'C{twisted.web.resource._computeAllowedMethods} is provided by a
    default should the subclass not provide the method.
    """

    def _getReq(self):
        """
        Generate a dummy request for use by C{_computeAllowedMethod} tests.
        """
        d = DummyChannel()
        d.site.resource.putChild(b'gettableresource', GettableResource())
        d.transport.port = 81
        request = server.Request(d, 1)
        request.setHost(b'example.com', 81)
        request.gotLength(0)
        return request

    def test_computeAllowedMethods(self):
        """
        C{_computeAllowedMethods} will search through the
        'gettableresource' for all attributes/methods of the form
        'render_{method}' ('render_GET', for example) and return a list of
        the methods. 'HEAD' will always be included from the
        resource.Resource superclass.
        """
        res = GettableResource()
        allowedMethods = resource._computeAllowedMethods(res)
        self.assertEqual(set(allowedMethods), {b'GET', b'HEAD', b'fred_render_ethel'})

    def test_notAllowed(self):
        """
        When an unsupported method is requested, the default
        L{_computeAllowedMethods} method will be called to determine the
        allowed methods, and the HTTP 405 'Method Not Allowed' status will
        be returned with the allowed methods will be returned in the
        'Allow' header.
        """
        req = self._getReq()
        req.requestReceived(b'POST', b'/gettableresource', b'HTTP/1.0')
        self.assertEqual(req.code, 405)
        self.assertEqual(set(req.responseHeaders.getRawHeaders(b'allow')[0].split(b', ')), {b'GET', b'HEAD', b'fred_render_ethel'})

    def test_notAllowedQuoting(self):
        """
        When an unsupported method response is generated, an HTML message will
        be displayed.  That message should include a quoted form of the URI and,
        since that value come from a browser and shouldn't necessarily be
        trusted.
        """
        req = self._getReq()
        req.requestReceived(b'POST', b'/gettableresource?value=<script>bad', b'HTTP/1.0')
        self.assertEqual(req.code, 405)
        renderedPage = req.transport.written.getvalue()
        self.assertNotIn(b'<script>bad', renderedPage)
        self.assertIn(b'&lt;script&gt;bad', renderedPage)

    def test_notImplementedQuoting(self):
        """
        When an not-implemented method response is generated, an HTML message
        will be displayed.  That message should include a quoted form of the
        requested method, since that value come from a browser and shouldn't
        necessarily be trusted.
        """
        req = self._getReq()
        req.requestReceived(b'<style>bad', b'/gettableresource', b'HTTP/1.0')
        self.assertEqual(req.code, 501)
        renderedPage = req.transport.written.getvalue()
        self.assertNotIn(b'<style>bad', renderedPage)
        self.assertIn(b'&lt;style&gt;bad', renderedPage)