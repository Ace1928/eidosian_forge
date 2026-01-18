import base64
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.cred import error, portal
from twisted.cred.checkers import (
from twisted.cred.credentials import IUsernamePassword
from twisted.internet.address import IPv4Address
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import EventLoggingObserver
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial import unittest
from twisted.web._auth import basic, digest
from twisted.web._auth.basic import BasicCredentialFactory
from twisted.web._auth.wrapper import HTTPAuthSessionWrapper, UnauthorizedResource
from twisted.web.iweb import ICredentialFactory
from twisted.web.resource import IResource, Resource, getChildForRequest
from twisted.web.server import NOT_DONE_YET
from twisted.web.static import Data
from twisted.web.test.test_web import DummyRequest
class UnauthorizedResourceTests(RequestMixin, unittest.TestCase):
    """
    Tests for L{UnauthorizedResource}.
    """

    def test_getChildWithDefault(self):
        """
        An L{UnauthorizedResource} is every child of itself.
        """
        resource = UnauthorizedResource([])
        self.assertIdentical(resource.getChildWithDefault('foo', None), resource)
        self.assertIdentical(resource.getChildWithDefault('bar', None), resource)

    def _unauthorizedRenderTest(self, request):
        """
        Render L{UnauthorizedResource} for the given request object and verify
        that the response code is I{Unauthorized} and that a I{WWW-Authenticate}
        header is set in the response containing a challenge.
        """
        resource = UnauthorizedResource([BasicCredentialFactory('example.com')])
        request.render(resource)
        self.assertEqual(request.responseCode, 401)
        self.assertEqual(request.responseHeaders.getRawHeaders(b'www-authenticate'), [b'basic realm="example.com"'])

    def test_render(self):
        """
        L{UnauthorizedResource} renders with a 401 response code and a
        I{WWW-Authenticate} header and puts a simple unauthorized message
        into the response body.
        """
        request = self.makeRequest()
        self._unauthorizedRenderTest(request)
        self.assertEqual(b'Unauthorized', b''.join(request.written))

    def test_renderHEAD(self):
        """
        The rendering behavior of L{UnauthorizedResource} for a I{HEAD} request
        is like its handling of a I{GET} request, but no response body is
        written.
        """
        request = self.makeRequest(method=b'HEAD')
        self._unauthorizedRenderTest(request)
        self.assertEqual(b'', b''.join(request.written))

    def test_renderQuotesRealm(self):
        """
        The realm value included in the I{WWW-Authenticate} header set in
        the response when L{UnauthorizedResounrce} is rendered has quotes
        and backslashes escaped.
        """
        resource = UnauthorizedResource([BasicCredentialFactory('example\\"foo')])
        request = self.makeRequest()
        request.render(resource)
        self.assertEqual(request.responseHeaders.getRawHeaders(b'www-authenticate'), [b'basic realm="example\\\\\\"foo"'])

    def test_renderQuotesDigest(self):
        """
        The digest value included in the I{WWW-Authenticate} header
        set in the response when L{UnauthorizedResource} is rendered
        has quotes and backslashes escaped.
        """
        resource = UnauthorizedResource([digest.DigestCredentialFactory(b'md5', b'example\\"foo')])
        request = self.makeRequest()
        request.render(resource)
        authHeader = request.responseHeaders.getRawHeaders(b'www-authenticate')[0]
        self.assertIn(b'realm="example\\\\\\"foo"', authHeader)
        self.assertIn(b'hm="md5', authHeader)