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
class WebTestCase(TestCase):
    """Base TestCase with useful instance vars and utility functions."""
    _req_class: Type[HTTPGitRequest] = TestHTTPGitRequest

    def setUp(self):
        super().setUp()
        self._environ = {}
        self._req = self._req_class(self._environ, self._start_response, handlers=self._handlers())
        self._status = None
        self._headers = []
        self._output = BytesIO()

    def _start_response(self, status, headers):
        self._status = status
        self._headers = list(headers)
        return self._output.write

    def _handlers(self):
        return None

    def assertContentTypeEquals(self, expected):
        self.assertIn(('Content-Type', expected), self._headers)