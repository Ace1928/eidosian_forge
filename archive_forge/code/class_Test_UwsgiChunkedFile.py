import datetime
import gettext
import http.client as http
import os
import socket
from unittest import mock
import eventlet.patcher
import fixtures
from oslo_concurrency import processutils
from oslo_serialization import jsonutils
import routes
import webob
from glance.api.v2 import router as router_v2
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance import i18n
from glance.image_cache import prefetcher
from glance.tests import utils as test_utils
class Test_UwsgiChunkedFile(test_utils.BaseTestCase):

    def test_read_no_data(self):
        reader = wsgi._UWSGIChunkFile()
        wsgi.uwsgi = mock.MagicMock()
        self.addCleanup(_cleanup_uwsgi)

        def fake_read():
            return None
        wsgi.uwsgi.chunked_read = fake_read
        out = reader.read()
        self.assertEqual(out, b'')

    def test_read_data_no_length(self):
        reader = wsgi._UWSGIChunkFile()
        wsgi.uwsgi = mock.MagicMock()
        self.addCleanup(_cleanup_uwsgi)
        values = iter([b'a', b'b', b'c', None])

        def fake_read():
            return next(values)
        wsgi.uwsgi.chunked_read = fake_read
        out = reader.read()
        self.assertEqual(out, b'abc')

    def test_read_zero_length(self):
        reader = wsgi._UWSGIChunkFile()
        self.assertEqual(b'', reader.read(length=0))

    def test_read_data_length(self):
        reader = wsgi._UWSGIChunkFile()
        wsgi.uwsgi = mock.MagicMock()
        self.addCleanup(_cleanup_uwsgi)
        values = iter([b'a', b'b', b'c', None])

        def fake_read():
            return next(values)
        wsgi.uwsgi.chunked_read = fake_read
        out = reader.read(length=2)
        self.assertEqual(out, b'ab')

    def test_read_data_negative_length(self):
        reader = wsgi._UWSGIChunkFile()
        wsgi.uwsgi = mock.MagicMock()
        self.addCleanup(_cleanup_uwsgi)
        values = iter([b'a', b'b', b'c', None])

        def fake_read():
            return next(values)
        wsgi.uwsgi.chunked_read = fake_read
        out = reader.read(length=-2)
        self.assertEqual(out, b'abc')