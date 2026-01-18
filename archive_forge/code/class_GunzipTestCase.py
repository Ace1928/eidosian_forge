import gzip
import os
import re
from io import BytesIO
from typing import Type
from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore
from ..objects import Blob
from ..repo import BaseRepo, MemoryRepo
from ..server import DictBackend
from ..web import (
from .utils import make_object, make_tag
class GunzipTestCase(HTTPGitApplicationTestCase):
    __doc__ = 'TestCase for testing the GunzipFilter, ensuring the wsgi.input\n    is correctly decompressed and headers are corrected.\n    '
    example_text = __doc__.encode('ascii')

    def setUp(self):
        super().setUp()
        self._app = GunzipFilter(self._app)
        self._environ['HTTP_CONTENT_ENCODING'] = 'gzip'
        self._environ['REQUEST_METHOD'] = 'POST'

    def _get_zstream(self, text):
        zstream = BytesIO()
        zfile = gzip.GzipFile(fileobj=zstream, mode='wb')
        zfile.write(text)
        zfile.close()
        zlength = zstream.tell()
        zstream.seek(0)
        return (zstream, zlength)

    def _test_call(self, orig, zstream, zlength):
        self._add_handler(self._app.app)
        self.assertLess(zlength, len(orig))
        self.assertEqual(self._environ['HTTP_CONTENT_ENCODING'], 'gzip')
        self._environ['CONTENT_LENGTH'] = zlength
        self._environ['wsgi.input'] = zstream
        self._app(self._environ, None)
        buf = self._environ['wsgi.input']
        self.assertIsNot(buf, zstream)
        buf.seek(0)
        self.assertEqual(orig, buf.read())
        self.assertIs(None, self._environ.get('CONTENT_LENGTH'))
        self.assertNotIn('HTTP_CONTENT_ENCODING', self._environ)

    def test_call(self):
        self._test_call(self.example_text, *self._get_zstream(self.example_text))

    def test_call_no_seek(self):
        """This ensures that the gunzipping code doesn't require any methods on
        'wsgi.input' except for '.read()'.  (In particular, it shouldn't
        require '.seek()'. See https://github.com/jelmer/dulwich/issues/140.).
        """
        zstream, zlength = self._get_zstream(self.example_text)
        self._test_call(self.example_text, MinimalistWSGIInputStream(zstream.read()), zlength)

    def test_call_no_working_seek(self):
        """Similar to 'test_call_no_seek', but this time the methods are available
        (but defunct).  See https://github.com/jonashaag/klaus/issues/154.
        """
        zstream, zlength = self._get_zstream(self.example_text)
        self._test_call(self.example_text, MinimalistWSGIInputStream2(zstream.read()), zlength)