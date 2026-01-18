from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
class CORSPreflightRequestTest(CORSTestBase):
    """CORS Specification Section 6.2

    http://www.w3.org/TR/cors/#resource-preflight-requests
    """

    def setUp(self):
        super(CORSPreflightRequestTest, self).setUp()
        fixture = self.config_fixture
        fixture.load_raw_values(group='cors', allowed_origin='http://valid.example.com', allow_credentials='False', max_age='', expose_headers='', allow_methods='GET', allow_headers='')
        fixture.load_raw_values(group='cors.credentials', allowed_origin='http://creds.example.com', allow_credentials='True')
        fixture.load_raw_values(group='cors.exposed-headers', allowed_origin='http://headers.example.com', expose_headers='X-Header-1,X-Header-2', allow_headers='X-Header-1,X-Header-2')
        fixture.load_raw_values(group='cors.cached', allowed_origin='http://cached.example.com', max_age='3600')
        fixture.load_raw_values(group='cors.get-only', allowed_origin='http://get.example.com', allow_methods='GET')
        fixture.load_raw_values(group='cors.all-methods', allowed_origin='http://all.example.com', allow_methods='GET,PUT,POST,DELETE,HEAD')
        self.application = cors.CORS(test_application, self.config)

    def test_config_overrides(self):
        """Assert that the configuration options are properly registered."""
        gc = self.config.cors
        self.assertEqual(gc.allowed_origin, ['http://valid.example.com'])
        self.assertEqual(gc.allow_credentials, False)
        self.assertEqual(gc.expose_headers, [])
        self.assertIsNone(gc.max_age)
        self.assertEqual(gc.allow_methods, ['GET'])
        self.assertEqual(gc.allow_headers, [])
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
        self.assertEqual(ac.allow_methods, ['GET', 'PUT', 'POST', 'DELETE', 'HEAD'])
        self.assertEqual(gc.allow_headers, ac.allow_headers)

    def test_no_origin_header(self):
        """CORS Specification Section 6.2.1

        If the Origin header is not present terminate this set of steps. The
        request is outside the scope of this specification.
        """
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)

    def test_case_sensitive_origin(self):
        """CORS Specification Section 6.2.2

        If the value of the Origin header is not a case-sensitive match for
        any of the values in list of origins do not set any additional headers
        and terminate this set of steps.
        """
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://valid.example.com'
        request.headers['Access-Control-Request-Method'] = 'GET'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://valid.example.com', max_age=None, allow_methods='GET', allow_headers='', allow_credentials=None, expose_headers=None)
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://invalid.example.com'
        request.headers['Access-Control-Request-Method'] = 'GET'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://VALID.EXAMPLE.COM'
        request.headers['Access-Control-Request-Method'] = 'GET'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)

    def test_simple_header_response(self):
        """CORS Specification Section 3

        A header is said to be a simple header if the header field name is an
        ASCII case-insensitive match for Accept, Accept-Language, or
        Content-Language or if it is an ASCII case-insensitive match for
        Content-Type and the header field value media type (excluding
        parameters) is an ASCII case-insensitive match for
        application/x-www-form-urlencoded, multipart/form-data, or text/plain.

        NOTE: We are not testing the media type cases.
        """
        simple_headers = ','.join(['accept', 'accept-language', 'content-language', 'content-type'])
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://valid.example.com'
        request.headers['Access-Control-Request-Method'] = 'GET'
        request.headers['Access-Control-Request-Headers'] = simple_headers
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://valid.example.com', max_age=None, allow_methods='GET', allow_headers=simple_headers, allow_credentials=None, expose_headers=None)

    def test_no_request_method(self):
        """CORS Specification Section 6.2.3

        If there is no Access-Control-Request-Method header or if parsing
        failed, do not set any additional headers and terminate this set of
        steps. The request is outside the scope of this specification.
        """
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://get.example.com'
        request.headers['Access-Control-Request-Method'] = 'GET'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://get.example.com', max_age=None, allow_methods='GET', allow_headers=None, allow_credentials=None, expose_headers=None)
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://valid.example.com'
        request.headers['Access-Control-Request-Method'] = 'TEAPOT'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://valid.example.com'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)

    def test_invalid_method(self):
        """CORS Specification Section 6.2.3

        If method is not a case-sensitive match for any of the values in
        list of methods do not set any additional headers and terminate this
        set of steps.
        """
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://get.example.com'
        request.headers['Access-Control-Request-Method'] = 'get'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)

    def test_no_parse_request_headers(self):
        """CORS Specification Section 6.2.4

        If there are no Access-Control-Request-Headers headers let header
        field-names be the empty list.

        If parsing failed do not set any additional headers and terminate
        this set of steps. The request is outside the scope of this
        specification.
        """
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://headers.example.com'
        request.headers['Access-Control-Request-Method'] = 'GET'
        request.headers['Access-Control-Request-Headers'] = 'value with spaces'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)

    def test_no_request_headers(self):
        """CORS Specification Section 6.2.4

        If there are no Access-Control-Request-Headers headers let header
        field-names be the empty list.
        """
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://headers.example.com'
        request.headers['Access-Control-Request-Method'] = 'GET'
        request.headers['Access-Control-Request-Headers'] = ''
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://headers.example.com', max_age=None, allow_methods='GET', allow_headers=None, allow_credentials=None, expose_headers=None)

    def test_request_headers(self):
        """CORS Specification Section 6.2.4

        Let header field-names be the values as result of parsing the
        Access-Control-Request-Headers headers.

        If there are no Access-Control-Request-Headers headers let header
        field-names be the empty list.
        """
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://headers.example.com'
        request.headers['Access-Control-Request-Method'] = 'GET'
        request.headers['Access-Control-Request-Headers'] = 'X-Header-1,X-Header-2'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://headers.example.com', max_age=None, allow_methods='GET', allow_headers='X-Header-1,X-Header-2', allow_credentials=None, expose_headers=None)

    def test_request_headers_not_permitted(self):
        """CORS Specification Section 6.2.4, 6.2.6

        If there are no Access-Control-Request-Headers headers let header
        field-names be the empty list.

        If any of the header field-names is not a ASCII case-insensitive
        match for any of the values in list of headers do not set any
        additional headers and terminate this set of steps.
        """
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://headers.example.com'
        request.headers['Access-Control-Request-Method'] = 'GET'
        request.headers['Access-Control-Request-Headers'] = 'X-Not-Exposed,X-Never-Exposed'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)

    def test_credentials(self):
        """CORS Specification Section 6.2.7

        If the resource supports credentials add a single
        Access-Control-Allow-Origin header, with the value of the Origin header
        as value, and add a single Access-Control-Allow-Credentials header with
        the case-sensitive string "true" as value.

        Otherwise, add a single Access-Control-Allow-Origin header, with either
        the value of the Origin header or the string "*" as value.

        NOTE: We never use the "*" as origin.
        """
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://creds.example.com'
        request.headers['Access-Control-Request-Method'] = 'GET'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://creds.example.com', max_age=None, allow_methods='GET', allow_headers=None, allow_credentials='true', expose_headers=None)

    def test_optional_max_age(self):
        """CORS Specification Section 6.2.8

        Optionally add a single Access-Control-Max-Age header with as value
        the amount of seconds the user agent is allowed to cache the result of
        the request.
        """
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://cached.example.com'
        request.headers['Access-Control-Request-Method'] = 'GET'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://cached.example.com', max_age=3600, allow_methods='GET', allow_headers=None, allow_credentials=None, expose_headers=None)

    def test_allow_methods(self):
        """CORS Specification Section 6.2.9

        Add one or more Access-Control-Allow-Methods headers consisting of
        (a subset of) the list of methods.

        Since the list of methods can be unbounded, simply returning the method
        indicated by Access-Control-Request-Method (if supported) can be
        enough.
        """
        for method in ['GET', 'PUT', 'POST', 'DELETE']:
            request = webob.Request.blank('/')
            request.method = 'OPTIONS'
            request.headers['Origin'] = 'http://all.example.com'
            request.headers['Access-Control-Request-Method'] = method
            response = request.get_response(self.application)
            self.assertCORSResponse(response, status='200 OK', allow_origin='http://all.example.com', max_age=None, allow_methods=method, allow_headers=None, allow_credentials=None, expose_headers=None)
        for method in ['PUT', 'POST', 'DELETE']:
            request = webob.Request.blank('/')
            request.method = 'OPTIONS'
            request.headers['Origin'] = 'http://get.example.com'
            request.headers['Access-Control-Request-Method'] = method
            response = request.get_response(self.application)
            self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)

    def test_allow_headers(self):
        """CORS Specification Section 6.2.10

        Add one or more Access-Control-Allow-Headers headers consisting of
        (a subset of) the list of headers.

        If each of the header field-names is a simple header and none is
        Content-Type, this step may be skipped.

        If a header field name is a simple header and is not Content-Type, it
        is not required to be listed. Content-Type is to be listed as only a
        subset of its values makes it qualify as simple header.
        """
        requested_headers = 'Content-Type,X-Header-1,Cache-Control,Expires,Last-Modified,Pragma'
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://headers.example.com'
        request.headers['Access-Control-Request-Method'] = 'GET'
        request.headers['Access-Control-Request-Headers'] = requested_headers
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://headers.example.com', max_age=None, allow_methods='GET', allow_headers=requested_headers, allow_credentials=None, expose_headers=None)

    def test_application_options_response(self):
        """Assert that an application provided OPTIONS response is honored.

        If the underlying application, via middleware or other, provides a
        CORS response, its response should be honored.
        """
        test_origin = 'http://creds.example.com'
        request = webob.Request.blank('/server_cors')
        request.method = 'OPTIONS'
        request.headers['Origin'] = test_origin
        request.headers['Access-Control-Request-Method'] = 'GET'
        response = request.get_response(self.application)
        self.assertNotIn('Access-Control-Allow-Credentials', response.headers)
        self.assertEqual(test_origin, response.headers['Access-Control-Allow-Origin'])
        self.assertEqual('1', response.headers['X-Server-Generated-Response'])
        request = webob.Request.blank('/server_no_cors')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://get.example.com'
        request.headers['Access-Control-Request-Method'] = 'GET'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://get.example.com', max_age=None, allow_methods='GET', allow_headers=None, allow_credentials=None, expose_headers=None, has_content_type=True)