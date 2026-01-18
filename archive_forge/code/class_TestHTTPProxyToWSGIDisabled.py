from wsgiref import util
from oslotest import base as test_base
import webob
from oslo_middleware import http_proxy_to_wsgi
class TestHTTPProxyToWSGIDisabled(test_base.BaseTestCase):

    def setUp(self):
        super(TestHTTPProxyToWSGIDisabled, self).setUp()

        @webob.dec.wsgify()
        def fake_app(req):
            return util.application_uri(req.environ)
        self.middleware = http_proxy_to_wsgi.HTTPProxyToWSGI(fake_app)
        self.middleware.oslo_conf.set_override('enable_proxy_headers_parsing', False, group='oslo_middleware')
        self.request = webob.Request.blank('/foo/bar', method='POST')

    def test_no_headers(self):
        response = self.request.get_response(self.middleware)
        self.assertEqual(b'http://localhost:80/', response.body)

    def test_url_translate_ssl_has_no_effect(self):
        self.request.headers['X-Forwarded-Proto'] = 'https'
        self.request.headers['X-Forwarded-Host'] = 'example.com:123'
        response = self.request.get_response(self.middleware)
        self.assertEqual(b'http://localhost:80/', response.body)