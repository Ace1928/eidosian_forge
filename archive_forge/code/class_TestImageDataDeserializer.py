import http.client as http
import io
from unittest import mock
import uuid
from cursive import exception as cursive_exception
import glance_store
from glance_store._drivers import filesystem
from oslo_config import cfg
import webob
import glance.api.policy
import glance.api.v2.image_data
from glance.common import exception
from glance.common import wsgi
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class TestImageDataDeserializer(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImageDataDeserializer, self).setUp()
        self.deserializer = glance.api.v2.image_data.RequestDeserializer()

    def test_upload(self):
        request = unit_test_utils.get_fake_request()
        request.headers['Content-Type'] = 'application/octet-stream'
        request.body = b'YYY'
        request.headers['Content-Length'] = 3
        output = self.deserializer.upload(request)
        data = output.pop('data')
        self.assertEqual(b'YYY', data.read())
        expected = {'size': 3}
        self.assertEqual(expected, output)

    def test_upload_chunked(self):
        request = unit_test_utils.get_fake_request()
        request.headers['Content-Type'] = 'application/octet-stream'
        request.body_file = io.StringIO('YYY')
        output = self.deserializer.upload(request)
        data = output.pop('data')
        self.assertEqual('YYY', data.read())
        expected = {'size': None}
        self.assertEqual(expected, output)

    def test_upload_chunked_with_content_length(self):
        request = unit_test_utils.get_fake_request()
        request.headers['Content-Type'] = 'application/octet-stream'
        request.body_file = io.BytesIO(b'YYY')
        request.headers['Content-Length'] = 3
        output = self.deserializer.upload(request)
        data = output.pop('data')
        self.assertEqual(b'YYY', data.read())
        expected = {'size': 3}
        self.assertEqual(expected, output)

    def test_upload_with_incorrect_content_length(self):
        request = unit_test_utils.get_fake_request()
        request.headers['Content-Type'] = 'application/octet-stream'
        request.body = b'YYY'
        request.headers['Content-Length'] = 4
        output = self.deserializer.upload(request)
        data = output.pop('data')
        self.assertEqual(b'YYY', data.read())
        expected = {'size': 4}
        self.assertEqual(expected, output)

    def test_upload_wrong_content_type(self):
        request = unit_test_utils.get_fake_request()
        request.headers['Content-Type'] = 'application/json'
        request.body = b'YYYYY'
        self.assertRaises(webob.exc.HTTPUnsupportedMediaType, self.deserializer.upload, request)
        request = unit_test_utils.get_fake_request()
        request.headers['Content-Type'] = 'application/octet-st'
        request.body = b'YYYYY'
        self.assertRaises(webob.exc.HTTPUnsupportedMediaType, self.deserializer.upload, request)

    def test_stage(self):
        req = unit_test_utils.get_fake_request(roles=['admin', 'member'])
        req.headers['Content-Type'] = 'application/octet-stream'
        req.headers['Content-Length'] = 4
        req.body_file = io.BytesIO(b'YYYY')
        output = self.deserializer.stage(req)
        data = output.pop('data')
        self.assertEqual(b'YYYY', data.read())

    def test_stage_without_glance_direct(self):
        self.config(enabled_import_methods=['web-download'])
        req = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPNotFound, self.deserializer.stage, req)

    def test_stage_raises_invalid_content_type(self):
        req = unit_test_utils.get_fake_request()
        req.headers['Content-Type'] = 'application/json'
        self.assertRaises(webob.exc.HTTPUnsupportedMediaType, self.deserializer.stage, req)