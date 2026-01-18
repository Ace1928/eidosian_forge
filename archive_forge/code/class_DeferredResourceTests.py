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
class DeferredResourceTests(SynchronousTestCase):
    """
    Tests for L{DeferredResource}.
    """

    def testDeferredResource(self):
        r = resource.Resource()
        r.isLeaf = 1
        s = SDResource(r)
        d = DummyRequest(['foo', 'bar', 'baz'])
        resource.getChildForRequest(s, d)
        self.assertEqual(d.postpath, ['bar', 'baz'])

    def test_render(self):
        """
        L{DeferredResource} uses the request object's C{render} method to
        render the resource which is the result of the L{Deferred} being
        handled.
        """
        rendered = []
        request = DummyRequest([])
        request.render = rendered.append
        result = resource.Resource()
        deferredResource = DeferredResource(defer.succeed(result))
        deferredResource.render(request)
        self.assertEqual(rendered, [result])

    def test_renderNoFailure(self):
        """
        If the L{Deferred} fails, L{DeferredResource} reports the failure via
        C{processingFailed}, and does not cause an unhandled error to be
        logged.
        """
        request = DummyRequest([])
        d = request.notifyFinish()
        failure = Failure(RuntimeError())
        deferredResource = DeferredResource(defer.fail(failure))
        deferredResource.render(request)
        self.assertEqual(self.failureResultOf(d), failure)
        del deferredResource
        gc.collect()
        errors = self.flushLoggedErrors(RuntimeError)
        self.assertEqual(errors, [])