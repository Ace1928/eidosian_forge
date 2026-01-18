import http.client as http
import ddt
import webob
from oslo_serialization import jsonutils
from glance.api.middleware import version_negotiation
from glance.api import versions
from glance.tests.unit import base
class VersionNegotiationTest(base.IsolatedUnitTest):

    def setUp(self):
        super(VersionNegotiationTest, self).setUp()
        self.middleware = version_negotiation.VersionNegotiationFilter(None)

    def test_request_url_v2(self):
        request = webob.Request.blank('/v2/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_0(self):
        request = webob.Request.blank('/v2.0/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_1(self):
        request = webob.Request.blank('/v2.1/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_2(self):
        request = webob.Request.blank('/v2.2/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_3(self):
        request = webob.Request.blank('/v2.3/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_4(self):
        request = webob.Request.blank('/v2.4/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_5(self):
        request = webob.Request.blank('/v2.5/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_6(self):
        request = webob.Request.blank('/v2.6/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_7(self):
        request = webob.Request.blank('/v2.7/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_9(self):
        request = webob.Request.blank('/v2.9/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_15(self):
        request = webob.Request.blank('/v2.15/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_8_default_unsupported(self):
        request = webob.Request.blank('/v2.8/images')
        resp = self.middleware.process_request(request)
        self.assertIsInstance(resp, versions.Controller)

    def test_request_url_v2_8_enabled_supported(self):
        self.config(enabled_backends='slow:one,fast:two')
        request = webob.Request.blank('/v2.8/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_10_default_unsupported(self):
        request = webob.Request.blank('/v2.10/images')
        resp = self.middleware.process_request(request)
        self.assertIsInstance(resp, versions.Controller)

    def test_request_url_v2_10_enabled_supported(self):
        self.config(enabled_backends='slow:one,fast:two')
        request = webob.Request.blank('/v2.10/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_11_default_unsupported(self):
        request = webob.Request.blank('/v2.11/images')
        resp = self.middleware.process_request(request)
        self.assertIsInstance(resp, versions.Controller)

    def test_request_url_v2_11_enabled_supported(self):
        self.config(enabled_backends='slow:one,fast:two')
        request = webob.Request.blank('/v2.11/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_12_default_unsupported(self):
        request = webob.Request.blank('/v2.12/images')
        resp = self.middleware.process_request(request)
        self.assertIsInstance(resp, versions.Controller)

    def test_request_url_v2_12_enabled_supported(self):
        self.config(enabled_backends='slow:one,fast:two')
        request = webob.Request.blank('/v2.12/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_13_default_unsupported(self):
        request = webob.Request.blank('/v2.13/images')
        resp = self.middleware.process_request(request)
        self.assertIsInstance(resp, versions.Controller)

    def test_request_url_v2_13_enabled_supported(self):
        self.config(enabled_backends='slow:one,fast:two')
        request = webob.Request.blank('/v2.13/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_14_default_unsupported(self):
        request = webob.Request.blank('/v2.14/images')
        resp = self.middleware.process_request(request)
        self.assertIsInstance(resp, versions.Controller)

    def test_request_url_v2_14_enabled_supported(self):
        self.config(image_cache_dir='/tmp/cache')
        request = webob.Request.blank('/v2.14/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_16_default_unsupported(self):
        request = webob.Request.blank('/v2.16/images')
        resp = self.middleware.process_request(request)
        self.assertIsInstance(resp, versions.Controller)

    def test_request_url_v2_16_enabled_supported(self):
        self.config(image_cache_dir='/tmp/cache')
        request = webob.Request.blank('/v2.16/images')
        self.middleware.process_request(request)
        self.assertEqual('/v2/images', request.path_info)

    def test_request_url_v2_17_default_unsupported(self):
        request = webob.Request.blank('/v2.17/images')
        resp = self.middleware.process_request(request)
        self.assertIsInstance(resp, versions.Controller)

    def test_request_url_v2_17_enabled_unsupported(self):
        self.config(enabled_backends='slow:one,fast:two')
        request = webob.Request.blank('/v2.17/images')
        resp = self.middleware.process_request(request)
        self.assertIsInstance(resp, versions.Controller)