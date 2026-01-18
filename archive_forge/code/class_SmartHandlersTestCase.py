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
class SmartHandlersTestCase(WebTestCase):

    class _TestUploadPackHandler:

        def __init__(self, backend, args, proto, stateless_rpc=None, advertise_refs=False) -> None:
            self.args = args
            self.proto = proto
            self.stateless_rpc = stateless_rpc
            self.advertise_refs = advertise_refs

        def handle(self):
            self.proto.write(b'handled input: ' + self.proto.recv(1024))

    def _make_handler(self, *args, **kwargs):
        self._handler = self._TestUploadPackHandler(*args, **kwargs)
        return self._handler

    def _handlers(self):
        return {b'git-upload-pack': self._make_handler}

    def test_handle_service_request_unknown(self):
        mat = re.search('.*', '/git-evil-handler')
        content = list(handle_service_request(self._req, 'backend', mat))
        self.assertEqual(HTTP_FORBIDDEN, self._status)
        self.assertNotIn(b'git-evil-handler', b''.join(content))
        self.assertFalse(self._req.cached)

    def _run_handle_service_request(self, content_length=None):
        self._environ['wsgi.input'] = BytesIO(b'foo')
        if content_length is not None:
            self._environ['CONTENT_LENGTH'] = content_length
        mat = re.search('.*', '/git-upload-pack')

        class Backend:

            def open_repository(self, path):
                return None
        handler_output = b''.join(handle_service_request(self._req, Backend(), mat))
        write_output = self._output.getvalue()
        self.assertEqual(b'', handler_output)
        self.assertEqual(b'handled input: foo', write_output)
        self.assertContentTypeEquals('application/x-git-upload-pack-result')
        self.assertFalse(self._handler.advertise_refs)
        self.assertTrue(self._handler.stateless_rpc)
        self.assertFalse(self._req.cached)

    def test_handle_service_request(self):
        self._run_handle_service_request()

    def test_handle_service_request_with_length(self):
        self._run_handle_service_request(content_length='3')

    def test_handle_service_request_empty_length(self):
        self._run_handle_service_request(content_length='')

    def test_get_info_refs_unknown(self):
        self._environ['QUERY_STRING'] = 'service=git-evil-handler'

        class Backend:

            def open_repository(self, url):
                return None
        mat = re.search('.*', '/git-evil-pack')
        content = list(get_info_refs(self._req, Backend(), mat))
        self.assertNotIn(b'git-evil-handler', b''.join(content))
        self.assertEqual(HTTP_FORBIDDEN, self._status)
        self.assertFalse(self._req.cached)

    def test_get_info_refs(self):
        self._environ['wsgi.input'] = BytesIO(b'foo')
        self._environ['QUERY_STRING'] = 'service=git-upload-pack'

        class Backend:

            def open_repository(self, url):
                return None
        mat = re.search('.*', '/git-upload-pack')
        handler_output = b''.join(get_info_refs(self._req, Backend(), mat))
        write_output = self._output.getvalue()
        self.assertEqual(b'001e# service=git-upload-pack\n0000handled input: ', write_output)
        self.assertEqual(b'', handler_output)
        self.assertTrue(self._handler.advertise_refs)
        self.assertTrue(self._handler.stateless_rpc)
        self.assertFalse(self._req.cached)