from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
class CORSRegularRequestTest(CORSTestBase):
    """CORS Specification Section 6.1

    http://www.w3.org/TR/cors/#resource-requests
    """
    methods = ['POST', 'PUT', 'DELETE', 'GET', 'TRACE', 'HEAD']

    def setUp(self):
        """Setup the tests."""
        super(CORSRegularRequestTest, self).setUp()
        fixture = self.config_fixture
        fixture.load_raw_values(group='cors', allowed_origin='http://valid.example.com', allow_credentials='False', max_age='', expose_headers='', allow_methods='GET', allow_headers='')
        fixture.load_raw_values(group='cors.credentials', allowed_origin='http://creds.example.com', allow_credentials='True')
        fixture.load_raw_values(group='cors.exposed-headers', allowed_origin='http://headers.example.com', expose_headers='X-Header-1,X-Header-2', allow_headers='X-Header-1,X-Header-2')
        fixture.load_raw_values(group='cors.cached', allowed_origin='http://cached.example.com', max_age='3600')
        fixture.load_raw_values(group='cors.get-only', allowed_origin='http://get.example.com', allow_methods='GET')
        fixture.load_raw_values(group='cors.all-methods', allowed_origin='http://all.example.com', allow_methods='GET,PUT,POST,DELETE,HEAD')
        fixture.load_raw_values(group='cors.duplicate', allowed_origin='http://domain1.example.com,http://domain2.example.com')
        self.application = cors.CORS(test_application, self.config)

    def test_config_overrides(self):
        """Assert that the configuration options are properly registered."""
        gc = self.config.cors
        self.assertEqual(['http://valid.example.com'], gc.allowed_origin)
        self.assertEqual(False, gc.allow_credentials)
        self.assertEqual([], gc.expose_headers)
        self.assertIsNone(gc.max_age)
        self.assertEqual(['GET'], gc.allow_methods)
        self.assertEqual([], gc.allow_headers)
        cc = self.config['cors.credentials']
        self.assertEqual(['http://creds.example.com'], cc.allowed_origin)
        self.assertEqual(True, cc.allow_credentials)
        self.assertEqual(gc.expose_headers, cc.expose_headers)
        self.assertEqual(gc.max_age, cc.max_age)
        self.assertEqual(gc.allow_methods, cc.allow_methods)
        self.assertEqual(gc.allow_headers, cc.allow_headers)
        ec = self.config['cors.exposed-headers']
        self.assertEqual(['http://headers.example.com'], ec.allowed_origin)
        self.assertEqual(gc.allow_credentials, ec.allow_credentials)
        self.assertEqual(['X-Header-1', 'X-Header-2'], ec.expose_headers)
        self.assertEqual(gc.max_age, ec.max_age)
        self.assertEqual(gc.allow_methods, ec.allow_methods)
        self.assertEqual(['X-Header-1', 'X-Header-2'], ec.allow_headers)
        chc = self.config['cors.cached']
        self.assertEqual(['http://cached.example.com'], chc.allowed_origin)
        self.assertEqual(gc.allow_credentials, chc.allow_credentials)
        self.assertEqual(gc.expose_headers, chc.expose_headers)
        self.assertEqual(3600, chc.max_age)
        self.assertEqual(gc.allow_methods, chc.allow_methods)
        self.assertEqual(gc.allow_headers, chc.allow_headers)
        goc = self.config['cors.get-only']
        self.assertEqual(['http://get.example.com'], goc.allowed_origin)
        self.assertEqual(gc.allow_credentials, goc.allow_credentials)
        self.assertEqual(gc.expose_headers, goc.expose_headers)
        self.assertEqual(gc.max_age, goc.max_age)
        self.assertEqual(['GET'], goc.allow_methods)
        self.assertEqual(gc.allow_headers, goc.allow_headers)
        ac = self.config['cors.all-methods']
        self.assertEqual(['http://all.example.com'], ac.allowed_origin)
        self.assertEqual(gc.allow_credentials, ac.allow_credentials)
        self.assertEqual(gc.expose_headers, ac.expose_headers)
        self.assertEqual(gc.max_age, ac.max_age)
        self.assertEqual(['GET', 'PUT', 'POST', 'DELETE', 'HEAD'], ac.allow_methods)
        self.assertEqual(gc.allow_headers, ac.allow_headers)
        ac = self.config['cors.duplicate']
        self.assertEqual(['http://domain1.example.com', 'http://domain2.example.com'], ac.allowed_origin)
        self.assertEqual(gc.allow_credentials, ac.allow_credentials)
        self.assertEqual(gc.expose_headers, ac.expose_headers)
        self.assertEqual(gc.max_age, ac.max_age)
        self.assertEqual(gc.allow_methods, ac.allow_methods)
        self.assertEqual(gc.allow_headers, ac.allow_headers)

    def test_no_origin_header(self):
        """CORS Specification Section 6.1.1

        If the Origin header is not present terminate this set of steps. The
        request is outside the scope of this specification.
        """
        for method in self.methods:
            request = webob.Request.blank('/')
            response = request.get_response(self.application)
            self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None, has_content_type=True)

    def test_origin_headers(self):
        """CORS Specification Section 6.1.2

        If the value of the Origin header is not a case-sensitive match for
        any of the values in list of origins, do not set any additional
        headers and terminate this set of steps.
        """
        for method in self.methods:
            request = webob.Request.blank('/')
            request.method = method
            request.headers['Origin'] = 'http://valid.example.com'
            response = request.get_response(self.application)
            self.assertCORSResponse(response, status='200 OK', allow_origin='http://valid.example.com', max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None, has_content_type=True)
        for method in self.methods:
            request = webob.Request.blank('/')
            request.method = method
            request.headers['Origin'] = 'http://invalid.example.com'
            response = request.get_response(self.application)
            self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None, has_content_type=True)
        for method in self.methods:
            request = webob.Request.blank('/')
            request.method = method
            request.headers['Origin'] = 'http://VALID.EXAMPLE.COM'
            response = request.get_response(self.application)
            self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None, has_content_type=True)
        for method in self.methods:
            request = webob.Request.blank('/')
            request.method = method
            request.headers['Origin'] = 'http://domain2.example.com'
            response = request.get_response(self.application)
            self.assertCORSResponse(response, status='200 OK', allow_origin='http://domain2.example.com', max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None, has_content_type=True)

    def test_supports_credentials(self):
        """CORS Specification Section 6.1.3

        If the resource supports credentials add a single
        Access-Control-Allow-Origin header, with the value of the Origin header
        as value, and add a single Access-Control-Allow-Credentials header with
        the case-sensitive string "true" as value.

        Otherwise, add a single Access-Control-Allow-Origin header, with
        either the value of the Origin header or the string "*" as value.

        NOTE: We never use the "*" as origin.
        """
        for method in self.methods:
            request = webob.Request.blank('/')
            request.method = method
            request.headers['Origin'] = 'http://valid.example.com'
            response = request.get_response(self.application)
            self.assertCORSResponse(response, status='200 OK', allow_origin='http://valid.example.com', max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None, has_content_type=True)
        for method in self.methods:
            request = webob.Request.blank('/')
            request.method = method
            request.headers['Origin'] = 'http://creds.example.com'
            response = request.get_response(self.application)
            self.assertCORSResponse(response, status='200 OK', allow_origin='http://creds.example.com', max_age=None, allow_methods=None, allow_headers=None, allow_credentials='true', expose_headers=None, has_content_type=True)

    def test_expose_headers(self):
        """CORS Specification Section 6.1.4

        If the list of exposed headers is not empty add one or more
        Access-Control-Expose-Headers headers, with as values the header field
        names given in the list of exposed headers.
        """
        for method in self.methods:
            request = webob.Request.blank('/')
            request.method = method
            request.headers['Origin'] = 'http://headers.example.com'
            response = request.get_response(self.application)
            self.assertCORSResponse(response, status='200 OK', allow_origin='http://headers.example.com', max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers='X-Header-1,X-Header-2', has_content_type=True)

    def test_application_options_response(self):
        """Assert that an application provided OPTIONS response is honored.

        If the underlying application, via middleware or other, provides a
        CORS response, its response should be honored.
        """
        test_origin = 'http://creds.example.com'
        request = webob.Request.blank('/server_cors')
        request.method = 'GET'
        request.headers['Origin'] = test_origin
        request.headers['Access-Control-Request-Method'] = 'GET'
        response = request.get_response(self.application)
        self.assertNotIn('Access-Control-Allow-Credentials', response.headers)
        self.assertEqual(response.headers['Access-Control-Allow-Origin'], test_origin)
        self.assertEqual(response.headers['X-Server-Generated-Response'], '1')

    def test_application_vary_respected(self):
        """Assert that an application's provided Vary header is persisted.

        If the underlying application, via middleware or other, provides a
        Vary header, its response should be honored.
        """
        request = webob.Request.blank('/server_cors_vary')
        request.method = 'GET'
        request.headers['Origin'] = 'http://valid.example.com'
        request.headers['Access-Control-Request-Method'] = 'GET'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://valid.example.com', max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None, vary='Custom-Vary,Origin', has_content_type=True)