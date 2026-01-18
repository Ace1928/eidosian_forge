import gc
from twisted.internet import defer
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import resource, util
from twisted.web.error import FlattenerError
from twisted.web.http import FOUND
from twisted.web.server import Request
from twisted.web.template import TagLoader, flattenString, tags
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from twisted.web.util import (
class ParentRedirectTests(SynchronousTestCase):
    """
    Test L{ParentRedirect}.
    """

    def doLocationTest(self, requestPath: bytes) -> bytes:
        """
        Render a response to a request with path *requestPath*

        @param requestPath: A slash-separated path like C{b'/foo/bar'}.

        @returns: The value of the I{Location} header.
        """
        request = Request(DummyChannel(), True)
        request.method = b'GET'
        request.prepath = requestPath.lstrip(b'/').split(b'/')
        resource = ParentRedirect()
        resource.render(request)
        headers = request.responseHeaders.getRawHeaders(b'Location')
        assert headers is not None
        [location] = headers
        return location

    def test_locationRoot(self):
        """
        At the URL root issue a redirect to the current URL, removing any query
        string.
        """
        self.assertEqual(b'http://10.0.0.1/', self.doLocationTest(b'/'))
        self.assertEqual(b'http://10.0.0.1/', self.doLocationTest(b'/?biff=baff'))

    def test_locationToRoot(self):
        """
        A request for a resource one level down from the URL root produces
        a redirect to the root.
        """
        self.assertEqual(b'http://10.0.0.1/', self.doLocationTest(b'/foo'))
        self.assertEqual(b'http://10.0.0.1/', self.doLocationTest(b'/foo?bar=sproiiing'))

    def test_locationUpOne(self):
        """
        Requests for resources directly under the path C{/foo/} produce
        redirects to C{/foo/}.
        """
        self.assertEqual(b'http://10.0.0.1/foo/', self.doLocationTest(b'/foo/'))
        self.assertEqual(b'http://10.0.0.1/foo/', self.doLocationTest(b'/foo/bar'))
        self.assertEqual(b'http://10.0.0.1/foo/', self.doLocationTest(b'/foo/bar?biz=baz'))