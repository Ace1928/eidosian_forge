import errno
import inspect
import mimetypes
import os
import re
import sys
import warnings
from io import BytesIO as StringIO
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import abstract, interfaces
from twisted.python import compat, log
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.web import http, resource, script, static
from twisted.web._responses import FOUND
from twisted.web.server import UnsupportedMethod
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyRequest
class StaticFileTests(TestCase):
    """
    Tests for the basic behavior of L{File}.
    """

    def _render(self, resource, request):
        return _render(resource, request)

    def test_ignoredExtTrue(self):
        """
        Passing C{1} as the value to L{File}'s C{ignoredExts} argument
        issues a warning and sets the ignored extensions to the
        wildcard C{"*"}.
        """
        with warnings.catch_warnings(record=True) as caughtWarnings:
            file = static.File(self.mktemp(), ignoredExts=1)
            self.assertEqual(file.ignoredExts, ['*'])
        self.assertEqual(len(caughtWarnings), 1)

    def test_ignoredExtFalse(self):
        """
        Passing C{1} as the value to L{File}'s C{ignoredExts} argument
        issues a warning and sets the ignored extensions to the empty
        list.
        """
        with warnings.catch_warnings(record=True) as caughtWarnings:
            file = static.File(self.mktemp(), ignoredExts=0)
            self.assertEqual(file.ignoredExts, [])
        self.assertEqual(len(caughtWarnings), 1)

    def test_allowExt(self):
        """
        Passing C{1} as the value to L{File}'s C{allowExt} argument
        issues a warning and sets the ignored extensions to the
        wildcard C{*}.
        """
        with warnings.catch_warnings(record=True) as caughtWarnings:
            file = static.File(self.mktemp(), ignoredExts=True)
            self.assertEqual(file.ignoredExts, ['*'])
        self.assertEqual(len(caughtWarnings), 1)

    def test_invalidMethod(self):
        """
        L{File.render} raises L{UnsupportedMethod} in response to a non-I{GET},
        non-I{HEAD} request.
        """
        request = DummyRequest([b''])
        request.method = b'POST'
        path = FilePath(self.mktemp())
        path.setContent(b'foo')
        file = static.File(path.path)
        self.assertRaises(UnsupportedMethod, file.render, request)

    def test_notFound(self):
        """
        If a request is made which encounters a L{File} before a final segment
        which does not correspond to any file in the path the L{File} was
        created with, a not found response is sent.
        """
        base = FilePath(self.mktemp())
        base.makedirs()
        file = static.File(base.path)
        request = DummyRequest([b'foobar'])
        child = resource.getChildForRequest(file, request)
        d = self._render(child, request)

        def cbRendered(ignored):
            self.assertEqual(request.responseCode, 404)
        d.addCallback(cbRendered)
        return d

    def test_emptyChild(self):
        """
        The C{''} child of a L{File} which corresponds to a directory in the
        filesystem is a L{DirectoryLister}.
        """
        base = FilePath(self.mktemp())
        base.makedirs()
        file = static.File(base.path)
        request = DummyRequest([b''])
        child = resource.getChildForRequest(file, request)
        self.assertIsInstance(child, static.DirectoryLister)
        self.assertEqual(child.path, base.path)

    def test_emptyChildUnicodeParent(self):
        """
        The C{u''} child of a L{File} which corresponds to a directory
        whose path is text is a L{DirectoryLister} that renders to a
        binary listing.

        @see: U{https://twistedmatrix.com/trac/ticket/9438}
        """
        textBase = FilePath(self.mktemp()).asTextMode()
        textBase.makedirs()
        textBase.child('text-file').open('w').close()
        textFile = static.File(textBase.path)
        request = DummyRequest([b''])
        child = resource.getChildForRequest(textFile, request)
        self.assertIsInstance(child, static.DirectoryLister)
        nativePath = compat.nativeString(textBase.path)
        self.assertEqual(child.path, nativePath)
        response = child.render(request)
        self.assertIsInstance(response, bytes)

    def test_securityViolationNotFound(self):
        """
        If a request is made which encounters a L{File} before a final segment
        which cannot be looked up in the filesystem due to security
        considerations, a not found response is sent.
        """
        base = FilePath(self.mktemp())
        base.makedirs()
        file = static.File(base.path)
        request = DummyRequest([b'..'])
        child = resource.getChildForRequest(file, request)
        d = self._render(child, request)

        def cbRendered(ignored):
            self.assertEqual(request.responseCode, 404)
        d.addCallback(cbRendered)
        return d

    @skipIf(platform.isWindows(), 'Cannot remove read permission on Windows')
    def test_forbiddenResource(self):
        """
        If the file in the filesystem which would satisfy a request cannot be
        read, L{File.render} sets the HTTP response code to I{FORBIDDEN}.
        """
        base = FilePath(self.mktemp())
        base.setContent(b'')
        self.addCleanup(base.chmod, 448)
        base.chmod(0)
        file = static.File(base.path)
        request = DummyRequest([b''])
        d = self._render(file, request)

        def cbRendered(ignored):
            self.assertEqual(request.responseCode, 403)
        d.addCallback(cbRendered)
        return d

    def test_undecodablePath(self):
        """
        A request whose path cannot be decoded as UTF-8 receives a not
        found response, and the failure is logged.
        """
        path = self.mktemp()
        if isinstance(path, bytes):
            path = path.decode('ascii')
        base = FilePath(path)
        base.makedirs()
        file = static.File(base.path)
        request = DummyRequest([b'\xff'])
        child = resource.getChildForRequest(file, request)
        d = self._render(child, request)

        def cbRendered(ignored):
            self.assertEqual(request.responseCode, 404)
            self.assertEqual(len(self.flushLoggedErrors(UnicodeDecodeError)), 1)
        d.addCallback(cbRendered)
        return d

    def test_forbiddenResource_default(self):
        """
        L{File.forbidden} defaults to L{resource.ForbiddenResource}.
        """
        self.assertIsInstance(static.File(b'.').forbidden, resource.ForbiddenResource)

    def test_forbiddenResource_customize(self):
        """
        The resource rendered for forbidden requests is stored as a class
        member so that users can customize it.
        """
        base = FilePath(self.mktemp())
        base.setContent(b'')
        markerResponse = b'custom-forbidden-response'

        def failingOpenForReading():
            raise OSError(errno.EACCES, '')

        class CustomForbiddenResource(resource.Resource):

            def render(self, request):
                return markerResponse

        class CustomStaticFile(static.File):
            forbidden = CustomForbiddenResource()
        fileResource = CustomStaticFile(base.path)
        fileResource.openForReading = failingOpenForReading
        request = DummyRequest([b''])
        result = fileResource.render(request)
        self.assertEqual(markerResponse, result)

    def test_indexNames(self):
        """
        If a request is made which encounters a L{File} before a final empty
        segment, a file in the L{File} instance's C{indexNames} list which
        exists in the path the L{File} was created with is served as the
        response to the request.
        """
        base = FilePath(self.mktemp())
        base.makedirs()
        base.child('foo.bar').setContent(b'baz')
        file = static.File(base.path)
        file.indexNames = ['foo.bar']
        request = DummyRequest([b''])
        child = resource.getChildForRequest(file, request)
        d = self._render(child, request)

        def cbRendered(ignored):
            self.assertEqual(b''.join(request.written), b'baz')
            self.assertEqual(request.responseHeaders.getRawHeaders(b'content-length')[0], b'3')
        d.addCallback(cbRendered)
        return d

    def test_staticFile(self):
        """
        If a request is made which encounters a L{File} before a final segment
        which names a file in the path the L{File} was created with, that file
        is served as the response to the request.
        """
        base = FilePath(self.mktemp())
        base.makedirs()
        base.child('foo.bar').setContent(b'baz')
        file = static.File(base.path)
        request = DummyRequest([b'foo.bar'])
        child = resource.getChildForRequest(file, request)
        d = self._render(child, request)

        def cbRendered(ignored):
            self.assertEqual(b''.join(request.written), b'baz')
            self.assertEqual(request.responseHeaders.getRawHeaders(b'content-length')[0], b'3')
        d.addCallback(cbRendered)
        return d

    @skipIf(sys.getfilesystemencoding().lower() not in ('utf-8', 'mcbs'), 'Cannot write unicode filenames with file system encoding of {}'.format(sys.getfilesystemencoding()))
    def test_staticFileUnicodeFileName(self):
        """
        A request for a existing unicode file path encoded as UTF-8
        returns the contents of that file.
        """
        name = 'ῆ'
        content = b'content'
        base = FilePath(self.mktemp())
        base.makedirs()
        base.child(name).setContent(content)
        file = static.File(base.path)
        request = DummyRequest([name.encode('utf-8')])
        child = resource.getChildForRequest(file, request)
        d = self._render(child, request)

        def cbRendered(ignored):
            self.assertEqual(b''.join(request.written), content)
            self.assertEqual(request.responseHeaders.getRawHeaders(b'content-length')[0], networkString(str(len(content))))
        d.addCallback(cbRendered)
        return d

    def test_staticFileDeletedGetChild(self):
        """
        A L{static.File} created for a directory which does not exist should
        return childNotFound from L{static.File.getChild}.
        """
        staticFile = static.File(self.mktemp())
        request = DummyRequest([b'foo.bar'])
        child = staticFile.getChild(b'foo.bar', request)
        self.assertEqual(child, staticFile.childNotFound)

    def test_staticFileDeletedRender(self):
        """
        A L{static.File} created for a file which does not exist should render
        its C{childNotFound} page.
        """
        staticFile = static.File(self.mktemp())
        request = DummyRequest([b'foo.bar'])
        request2 = DummyRequest([b'foo.bar'])
        d = self._render(staticFile, request)
        d2 = self._render(staticFile.childNotFound, request2)

        def cbRendered2(ignored):

            def cbRendered(ignored):
                self.assertEqual(b''.join(request.written), b''.join(request2.written))
            d.addCallback(cbRendered)
            return d
        d2.addCallback(cbRendered2)
        return d2

    def test_getChildChildNotFound_customize(self):
        """
        The resource rendered for child not found requests can be customize
        using a class member.
        """
        base = FilePath(self.mktemp())
        base.setContent(b'')
        markerResponse = b'custom-child-not-found-response'

        class CustomChildNotFoundResource(resource.Resource):

            def render(self, request):
                return markerResponse

        class CustomStaticFile(static.File):
            childNotFound = CustomChildNotFoundResource()
        fileResource = CustomStaticFile(base.path)
        request = DummyRequest([b'no-child.txt'])
        child = fileResource.getChild(b'no-child.txt', request)
        result = child.render(request)
        self.assertEqual(markerResponse, result)

    def test_headRequest(self):
        """
        L{static.File.render} returns an empty response body for I{HEAD}
        requests.
        """
        path = FilePath(self.mktemp())
        path.setContent(b'foo')
        file = static.File(path.path)
        request = DummyRequest([b''])
        request.method = b'HEAD'
        d = _render(file, request)

        def cbRendered(ignored):
            self.assertEqual(b''.join(request.written), b'')
        d.addCallback(cbRendered)
        return d

    def test_processors(self):
        """
        If a request is made which encounters a L{File} before a final segment
        which names a file with an extension which is in the L{File}'s
        C{processors} mapping, the processor associated with that extension is
        used to serve the response to the request.
        """
        base = FilePath(self.mktemp())
        base.makedirs()
        base.child('foo.bar').setContent(b"from twisted.web.static import Data\nresource = Data(b'dynamic world', 'text/plain')\n")
        file = static.File(base.path)
        file.processors = {'.bar': script.ResourceScript}
        request = DummyRequest([b'foo.bar'])
        child = resource.getChildForRequest(file, request)
        d = self._render(child, request)

        def cbRendered(ignored):
            self.assertEqual(b''.join(request.written), b'dynamic world')
            self.assertEqual(request.responseHeaders.getRawHeaders(b'content-length')[0], b'13')
        d.addCallback(cbRendered)
        return d

    def test_ignoreExt(self):
        """
        The list of ignored extensions can be set by passing a value to
        L{File.__init__} or by calling L{File.ignoreExt} later.
        """
        file = static.File(b'.')
        self.assertEqual(file.ignoredExts, [])
        file.ignoreExt('.foo')
        file.ignoreExt('.bar')
        self.assertEqual(file.ignoredExts, ['.foo', '.bar'])
        file = static.File(b'.', ignoredExts=('.bar', '.baz'))
        self.assertEqual(file.ignoredExts, ['.bar', '.baz'])

    def test_ignoredExtensionsIgnored(self):
        """
        A request for the I{base} child of a L{File} succeeds with a resource
        for the I{base<extension>} file in the path the L{File} was created
        with if such a file exists and the L{File} has been configured to
        ignore the I{<extension>} extension.
        """
        base = FilePath(self.mktemp())
        base.makedirs()
        base.child('foo.bar').setContent(b'baz')
        base.child('foo.quux').setContent(b'foobar')
        file = static.File(base.path, ignoredExts=('.bar',))
        request = DummyRequest([b'foo'])
        child = resource.getChildForRequest(file, request)
        d = self._render(child, request)

        def cbRendered(ignored):
            self.assertEqual(b''.join(request.written), b'baz')
        d.addCallback(cbRendered)
        return d

    def test_directoryWithoutTrailingSlashRedirects(self):
        """
        A request for a path which is a directory but does not have a trailing
        slash will be redirected to a URL which does have a slash by L{File}.
        """
        base = FilePath(self.mktemp())
        base.makedirs()
        base.child('folder').makedirs()
        file = static.File(base.path)
        request = DummyRequest([b'folder'])
        request.uri = b'http://dummy/folder#baz?foo=bar'
        child = resource.getChildForRequest(file, request)
        self.successResultOf(self._render(child, request))
        self.assertEqual(request.responseCode, FOUND)
        self.assertEqual(request.responseHeaders.getRawHeaders(b'location'), [b'http://dummy/folder/#baz?foo=bar'])

    def _makeFilePathWithStringIO(self):
        """
        Create a L{File} that when opened for reading, returns a L{StringIO}.

        @return: 2-tuple of the opened "file" and the L{File}.
        @rtype: L{tuple}
        """
        fakeFile = StringIO()
        path = FilePath(self.mktemp())
        path.touch()
        file = static.File(path.path)
        file.open = lambda: fakeFile
        return (fakeFile, file)

    def test_HEADClosesFile(self):
        """
        A HEAD request opens the file, gets the size, and then closes it after
        the request.
        """
        fakeFile, file = self._makeFilePathWithStringIO()
        request = DummyRequest([''])
        request.method = b'HEAD'
        self.successResultOf(_render(file, request))
        self.assertEqual(b''.join(request.written), b'')
        self.assertTrue(fakeFile.closed)

    def test_cachedRequestClosesFile(self):
        """
        A GET request that is cached closes the file after the request.
        """
        fakeFile, file = self._makeFilePathWithStringIO()
        request = DummyRequest([''])
        request.method = b'GET'
        request.setLastModified = lambda _: http.CACHED
        self.successResultOf(_render(file, request))
        self.assertEqual(b''.join(request.written), b'')
        self.assertTrue(fakeFile.closed)