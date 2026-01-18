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
class DirectoryListerTests(TestCase):
    """
    Tests for L{static.DirectoryLister}.
    """

    def _request(self, uri):
        request = DummyRequest([b''])
        request.uri = uri
        return request

    def test_renderHeader(self):
        """
        L{static.DirectoryLister} prints the request uri as header of the
        rendered content.
        """
        path = FilePath(self.mktemp())
        path.makedirs()
        lister = static.DirectoryLister(path.path)
        data = lister.render(self._request(b'foo'))
        self.assertIn(b'<h1>Directory listing for foo</h1>', data)
        self.assertIn(b'<title>Directory listing for foo</title>', data)

    def test_renderUnquoteHeader(self):
        """
        L{static.DirectoryLister} unquote the request uri before printing it.
        """
        path = FilePath(self.mktemp())
        path.makedirs()
        lister = static.DirectoryLister(path.path)
        data = lister.render(self._request(b'foo%20bar'))
        self.assertIn(b'<h1>Directory listing for foo bar</h1>', data)
        self.assertIn(b'<title>Directory listing for foo bar</title>', data)

    def test_escapeHeader(self):
        """
        L{static.DirectoryLister} escape "&", "<" and ">" after unquoting the
        request uri.
        """
        path = FilePath(self.mktemp())
        path.makedirs()
        lister = static.DirectoryLister(path.path)
        data = lister.render(self._request(b'foo%26bar'))
        self.assertIn(b'<h1>Directory listing for foo&amp;bar</h1>', data)
        self.assertIn(b'<title>Directory listing for foo&amp;bar</title>', data)

    def test_renderFiles(self):
        """
        L{static.DirectoryLister} is able to list all the files inside a
        directory.
        """
        path = FilePath(self.mktemp())
        path.makedirs()
        path.child('file1').setContent(b'content1')
        path.child('file2').setContent(b'content2' * 1000)
        lister = static.DirectoryLister(path.path)
        data = lister.render(self._request(b'foo'))
        body = b'<tr class="odd">\n    <td><a href="file1">file1</a></td>\n    <td>8B</td>\n    <td>[text/html]</td>\n    <td></td>\n</tr>\n<tr class="even">\n    <td><a href="file2">file2</a></td>\n    <td>7K</td>\n    <td>[text/html]</td>\n    <td></td>\n</tr>'
        self.assertIn(body, data)

    def test_renderDirectories(self):
        """
        L{static.DirectoryLister} is able to list all the directories inside
        a directory.
        """
        path = FilePath(self.mktemp())
        path.makedirs()
        path.child('dir1').makedirs()
        path.child('dir2 & 3').makedirs()
        lister = static.DirectoryLister(path.path)
        data = lister.render(self._request(b'foo'))
        body = b'<tr class="odd">\n    <td><a href="dir1/">dir1/</a></td>\n    <td></td>\n    <td>[Directory]</td>\n    <td></td>\n</tr>\n<tr class="even">\n    <td><a href="dir2%20%26%203/">dir2 &amp; 3/</a></td>\n    <td></td>\n    <td>[Directory]</td>\n    <td></td>\n</tr>'
        self.assertIn(body, data)

    def test_renderFiltered(self):
        """
        L{static.DirectoryLister} takes an optional C{dirs} argument that
        filter out the list of directories and files printed.
        """
        path = FilePath(self.mktemp())
        path.makedirs()
        path.child('dir1').makedirs()
        path.child('dir2').makedirs()
        path.child('dir3').makedirs()
        lister = static.DirectoryLister(path.path, dirs=['dir1', 'dir3'])
        data = lister.render(self._request(b'foo'))
        body = b'<tr class="odd">\n    <td><a href="dir1/">dir1/</a></td>\n    <td></td>\n    <td>[Directory]</td>\n    <td></td>\n</tr>\n<tr class="even">\n    <td><a href="dir3/">dir3/</a></td>\n    <td></td>\n    <td>[Directory]</td>\n    <td></td>\n</tr>'
        self.assertIn(body, data)

    def test_oddAndEven(self):
        """
        L{static.DirectoryLister} gives an alternate class for each odd and
        even rows in the table.
        """
        lister = static.DirectoryLister(None)
        elements = [{'href': '', 'text': '', 'size': '', 'type': '', 'encoding': ''} for i in range(5)]
        content = lister._buildTableContent(elements)
        self.assertEqual(len(content), 5)
        self.assertTrue(content[0].startswith('<tr class="odd">'))
        self.assertTrue(content[1].startswith('<tr class="even">'))
        self.assertTrue(content[2].startswith('<tr class="odd">'))
        self.assertTrue(content[3].startswith('<tr class="even">'))
        self.assertTrue(content[4].startswith('<tr class="odd">'))

    def test_contentType(self):
        """
        L{static.DirectoryLister} produces a MIME-type that indicates that it is
        HTML, and includes its charset (UTF-8).
        """
        path = FilePath(self.mktemp())
        path.makedirs()
        lister = static.DirectoryLister(path.path)
        req = self._request(b'')
        lister.render(req)
        self.assertEqual(req.responseHeaders.getRawHeaders(b'content-type')[0], b'text/html; charset=utf-8')

    def test_mimeTypeAndEncodings(self):
        """
        L{static.DirectoryLister} is able to detect mimetype and encoding of
        listed files.
        """
        path = FilePath(self.mktemp())
        path.makedirs()
        path.child('file1.txt').setContent(b'file1')
        path.child('file2.py').setContent(b'python')
        path.child('file3.conf.gz').setContent(b'conf compressed')
        path.child('file4.diff.bz2').setContent(b'diff compressed')
        directory = os.listdir(path.path)
        directory.sort()
        contentTypes = {'.txt': 'text/plain', '.py': 'text/python', '.conf': 'text/configuration', '.diff': 'text/diff'}
        lister = static.DirectoryLister(path.path, contentTypes=contentTypes)
        dirs, files = lister._getFilesAndDirectories(directory)
        self.assertEqual(dirs, [])
        self.assertEqual(files, [{'encoding': '', 'href': 'file1.txt', 'size': '5B', 'text': 'file1.txt', 'type': '[text/plain]'}, {'encoding': '', 'href': 'file2.py', 'size': '6B', 'text': 'file2.py', 'type': '[text/python]'}, {'encoding': '[gzip]', 'href': 'file3.conf.gz', 'size': '15B', 'text': 'file3.conf.gz', 'type': '[text/configuration]'}, {'encoding': '[bzip2]', 'href': 'file4.diff.bz2', 'size': '15B', 'text': 'file4.diff.bz2', 'type': '[text/diff]'}])

    @skipIf(not platform._supportsSymlinks(), 'No symlink support')
    def test_brokenSymlink(self):
        """
        If on the file in the listing points to a broken symlink, it should not
        be returned by L{static.DirectoryLister._getFilesAndDirectories}.
        """
        path = FilePath(self.mktemp())
        path.makedirs()
        file1 = path.child('file1')
        file1.setContent(b'file1')
        file1.linkTo(path.child('file2'))
        file1.remove()
        lister = static.DirectoryLister(path.path)
        directory = os.listdir(path.path)
        directory.sort()
        dirs, files = lister._getFilesAndDirectories(directory)
        self.assertEqual(dirs, [])
        self.assertEqual(files, [])

    def test_childrenNotFound(self):
        """
        Any child resource of L{static.DirectoryLister} renders an HTTP
        I{NOT FOUND} response code.
        """
        path = FilePath(self.mktemp())
        path.makedirs()
        lister = static.DirectoryLister(path.path)
        request = self._request(b'')
        child = resource.getChildForRequest(lister, request)
        result = _render(child, request)

        def cbRendered(ignored):
            self.assertEqual(request.responseCode, http.NOT_FOUND)
        result.addCallback(cbRendered)
        return result

    def test_repr(self):
        """
        L{static.DirectoryLister.__repr__} gives the path of the lister.
        """
        path = FilePath(self.mktemp())
        lister = static.DirectoryLister(path.path)
        self.assertEqual(repr(lister), f'<DirectoryLister of {path.path!r}>')
        self.assertEqual(str(lister), f'<DirectoryLister of {path.path!r}>')

    def test_formatFileSize(self):
        """
        L{static.formatFileSize} format an amount of bytes into a more readable
        format.
        """
        self.assertEqual(static.formatFileSize(0), '0B')
        self.assertEqual(static.formatFileSize(123), '123B')
        self.assertEqual(static.formatFileSize(4567), '4K')
        self.assertEqual(static.formatFileSize(8900000), '8M')
        self.assertEqual(static.formatFileSize(1234000000), '1G')
        self.assertEqual(static.formatFileSize(1234567890000), '1149G')