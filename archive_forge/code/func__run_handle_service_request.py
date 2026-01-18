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