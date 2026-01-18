from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
class GetChildForRequestTests(TestCase):
    """
    Tests for L{getChildForRequest}.
    """

    def test_exhaustedPostPath(self) -> None:
        """
        L{getChildForRequest} returns whatever resource has been reached by the
        time the request's C{postpath} is empty.
        """
        request = DummyRequest([])
        resource = Resource()
        result = getChildForRequest(resource, request)
        self.assertIdentical(resource, result)

    def test_leafResource(self) -> None:
        """
        L{getChildForRequest} returns the first resource it encounters with a
        C{isLeaf} attribute set to C{True}.
        """
        request = DummyRequest([b'foo', b'bar'])
        resource = Resource()
        resource.isLeaf = True
        result = getChildForRequest(resource, request)
        self.assertIdentical(resource, result)

    def test_postPathToPrePath(self) -> None:
        """
        As path segments from the request are traversed, they are taken from
        C{postpath} and put into C{prepath}.
        """
        request = DummyRequest([b'foo', b'bar'])
        root = Resource()
        child = Resource()
        child.isLeaf = True
        root.putChild(b'foo', child)
        self.assertIdentical(child, getChildForRequest(root, request))
        self.assertEqual(request.prepath, [b'foo'])
        self.assertEqual(request.postpath, [b'bar'])