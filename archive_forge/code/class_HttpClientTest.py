from http import client as http_client
import io
from unittest import mock
from oslo_serialization import jsonutils
import socket
from magnumclient.common.apiclient.exceptions import GatewayTimeout
from magnumclient.common.apiclient.exceptions import MultipleChoices
from magnumclient.common import httpclient as http
from magnumclient import exceptions as exc
from magnumclient.tests import utils
class HttpClientTest(utils.BaseTestCase):

    def test_url_generation_trailing_slash_in_base(self):
        client = http.HTTPClient('http://localhost/')
        url = client._make_connection_url('/v1/resources')
        self.assertEqual('/v1/resources', url)

    def test_url_generation_without_trailing_slash_in_base(self):
        client = http.HTTPClient('http://localhost')
        url = client._make_connection_url('/v1/resources')
        self.assertEqual('/v1/resources', url)

    def test_url_generation_prefix_slash_in_path(self):
        client = http.HTTPClient('http://localhost/')
        url = client._make_connection_url('/v1/resources')
        self.assertEqual('/v1/resources', url)

    def test_url_generation_without_prefix_slash_in_path(self):
        client = http.HTTPClient('http://localhost')
        url = client._make_connection_url('v1/resources')
        self.assertEqual('/v1/resources', url)

    def test_server_exception_empty_body(self):
        error_body = _get_error_body()
        fake_resp = utils.FakeResponse({'content-type': 'application/json'}, io.StringIO(error_body), version=1, status=500)
        client = http.HTTPClient('http://localhost/')
        client.get_connection = lambda *a, **kw: utils.FakeConnection(fake_resp)
        error = self.assertRaises(exc.InternalServerError, client.json_request, 'GET', '/v1/resources')
        self.assertEqual('Internal Server Error (HTTP 500)', str(error))

    def test_server_exception_msg_only(self):
        error_msg = 'test error msg'
        error_body = _get_error_body(error_msg, err_type=ERROR_DICT)
        fake_resp = utils.FakeResponse({'content-type': 'application/json'}, io.StringIO(error_body), version=1, status=500)
        client = http.HTTPClient('http://localhost/')
        client.get_connection = lambda *a, **kw: utils.FakeConnection(fake_resp)
        error = self.assertRaises(exc.InternalServerError, client.json_request, 'GET', '/v1/resources')
        self.assertEqual(error_msg + ' (HTTP 500)', str(error))

    def test_server_exception_msg_and_traceback(self):
        error_msg = 'another test error'
        error_trace = '"Traceback (most recent call last):\\n\\n  File \\"/usr/local/lib/python2.7/...'
        error_body = _get_error_body(error_msg, error_trace, ERROR_LIST_WITH_DESC)
        fake_resp = utils.FakeResponse({'content-type': 'application/json'}, io.StringIO(error_body), version=1, status=500)
        client = http.HTTPClient('http://localhost/')
        client.get_connection = lambda *a, **kw: utils.FakeConnection(fake_resp)
        error = self.assertRaises(exc.InternalServerError, client.json_request, 'GET', '/v1/resources')
        self.assertEqual('%(error)s (HTTP 500)\n%(trace)s' % {'error': error_msg, 'trace': error_trace}, '%(error)s\n%(details)s' % {'error': str(error), 'details': str(error.details)})

    def test_server_exception_address(self):
        endpoint = 'https://magnum-host:6385'
        client = http.HTTPClient(endpoint, token='foobar', insecure=True, ca_file='/path/to/ca_file')
        client.get_connection = lambda *a, **kw: utils.FakeConnection(exc=socket.gaierror)
        self.assertRaises(exc.EndpointNotFound, client.json_request, 'GET', '/v1/resources', body='farboo')

    def test_server_exception_socket(self):
        client = http.HTTPClient('http://localhost/', token='foobar')
        client.get_connection = lambda *a, **kw: utils.FakeConnection(exc=socket.error)
        self.assertRaises(exc.ConnectionRefused, client.json_request, 'GET', '/v1/resources')

    def test_server_exception_endpoint(self):
        endpoint = 'https://magnum-host:6385'
        client = http.HTTPClient(endpoint, token='foobar', insecure=True, ca_file='/path/to/ca_file')
        client.get_connection = lambda *a, **kw: utils.FakeConnection(exc=socket.gaierror)
        self.assertRaises(exc.EndpointNotFound, client.json_request, 'GET', '/v1/resources', body='farboo')

    def test_get_connection(self):
        endpoint = 'https://magnum-host:6385'
        client = http.HTTPClient(endpoint)
        conn = client.get_connection()
        self.assertTrue(conn, http.VerifiedHTTPSConnection)

    def test_get_connection_exception(self):
        endpoint = 'http://magnum-host:6385/'
        expected = (HTTP_CLASS, ('magnum-host', 6385, ''), {'timeout': DEFAULT_TIMEOUT})
        params = http.HTTPClient.get_connection_params(endpoint)
        self.assertEqual(expected, params)

    def test_get_connection_params_with_ssl(self):
        endpoint = 'https://magnum-host:6385'
        expected = (HTTPS_CLASS, ('magnum-host', 6385, ''), {'timeout': DEFAULT_TIMEOUT, 'ca_file': None, 'cert_file': None, 'key_file': None, 'insecure': False})
        params = http.HTTPClient.get_connection_params(endpoint)
        self.assertEqual(expected, params)

    def test_get_connection_params_with_ssl_params(self):
        endpoint = 'https://magnum-host:6385'
        ssl_args = {'ca_file': '/path/to/ca_file', 'cert_file': '/path/to/cert_file', 'key_file': '/path/to/key_file', 'insecure': True}
        expected_kwargs = {'timeout': DEFAULT_TIMEOUT}
        expected_kwargs.update(ssl_args)
        expected = (HTTPS_CLASS, ('magnum-host', 6385, ''), expected_kwargs)
        params = http.HTTPClient.get_connection_params(endpoint, **ssl_args)
        self.assertEqual(expected, params)

    def test_get_connection_params_with_timeout(self):
        endpoint = 'http://magnum-host:6385'
        expected = (HTTP_CLASS, ('magnum-host', 6385, ''), {'timeout': 300.0})
        params = http.HTTPClient.get_connection_params(endpoint, timeout=300)
        self.assertEqual(expected, params)

    def test_get_connection_params_with_version(self):
        endpoint = 'http://magnum-host:6385/v1'
        expected = (HTTP_CLASS, ('magnum-host', 6385, ''), {'timeout': DEFAULT_TIMEOUT})
        params = http.HTTPClient.get_connection_params(endpoint)
        self.assertEqual(expected, params)

    def test_get_connection_params_with_version_trailing_slash(self):
        endpoint = 'http://magnum-host:6385/v1/'
        expected = (HTTP_CLASS, ('magnum-host', 6385, ''), {'timeout': DEFAULT_TIMEOUT})
        params = http.HTTPClient.get_connection_params(endpoint)
        self.assertEqual(expected, params)

    def test_get_connection_params_with_subpath(self):
        endpoint = 'http://magnum-host:6385/magnum'
        expected = (HTTP_CLASS, ('magnum-host', 6385, '/magnum'), {'timeout': DEFAULT_TIMEOUT})
        params = http.HTTPClient.get_connection_params(endpoint)
        self.assertEqual(expected, params)

    def test_get_connection_params_with_subpath_trailing_slash(self):
        endpoint = 'http://magnum-host:6385/magnum/'
        expected = (HTTP_CLASS, ('magnum-host', 6385, '/magnum'), {'timeout': DEFAULT_TIMEOUT})
        params = http.HTTPClient.get_connection_params(endpoint)
        self.assertEqual(expected, params)

    def test_get_connection_params_with_subpath_version(self):
        endpoint = 'http://magnum-host:6385/magnum/v1'
        expected = (HTTP_CLASS, ('magnum-host', 6385, '/magnum'), {'timeout': DEFAULT_TIMEOUT})
        params = http.HTTPClient.get_connection_params(endpoint)
        self.assertEqual(expected, params)

    def test_get_connection_params_with_subpath_version_trailing_slash(self):
        endpoint = 'http://magnum-host:6385/magnum/v1/'
        expected = (HTTP_CLASS, ('magnum-host', 6385, '/magnum'), {'timeout': DEFAULT_TIMEOUT})
        params = http.HTTPClient.get_connection_params(endpoint)
        self.assertEqual(expected, params)

    def test_get_connection_params_with_unsupported_scheme(self):
        endpoint = 'foo://magnum-host:6385/magnum/v1/'
        self.assertRaises(exc.EndpointException, http.HTTPClient.get_connection_params, endpoint)

    def test_401_unauthorized_exception(self):
        error_body = _get_error_body(err_type=ERROR_LIST_WITH_DETAIL)
        fake_resp = utils.FakeResponse({'content-type': 'text/plain'}, io.StringIO(error_body), version=1, status=401)
        client = http.HTTPClient('http://localhost/')
        client.get_connection = lambda *a, **kw: utils.FakeConnection(fake_resp)
        self.assertRaises(exc.Unauthorized, client.json_request, 'GET', '/v1/resources')

    def test_server_redirect_exception(self):
        fake_redirect_resp = utils.FakeResponse({'content-type': 'application/octet-stream'}, 'foo', version=1, status=301)
        fake_resp = utils.FakeResponse({'content-type': 'application/octet-stream'}, 'bar', version=1, status=300)
        client = http.HTTPClient('http://localhost/')
        conn = utils.FakeConnection(fake_redirect_resp, redirect_resp=fake_resp)
        client.get_connection = lambda *a, **kw: conn
        self.assertRaises(MultipleChoices, client.json_request, 'GET', '/v1/resources')

    def test_server_body_undecode_json(self):
        err = 'foo'
        fake_resp = utils.FakeResponse({'content-type': 'application/json'}, io.StringIO(err), version=1, status=200)
        client = http.HTTPClient('http://localhost/')
        conn = utils.FakeConnection(fake_resp)
        client.get_connection = lambda *a, **kw: conn
        resp, body = client.json_request('GET', '/v1/resources')
        self.assertEqual(resp, fake_resp)
        self.assertEqual(err, body)

    def test_server_success_body_app(self):
        fake_resp = utils.FakeResponse({'content-type': 'application/octet-stream'}, 'bar', version=1, status=200)
        client = http.HTTPClient('http://localhost/')
        conn = utils.FakeConnection(fake_resp)
        client.get_connection = lambda *a, **kw: conn
        resp, body = client.json_request('GET', '/v1/resources')
        self.assertEqual(resp, fake_resp)
        self.assertIsNone(body)

    def test_server_success_body_none(self):
        fake_resp = utils.FakeResponse({'content-type': None}, io.StringIO('bar'), version=1, status=200)
        client = http.HTTPClient('http://localhost/')
        conn = utils.FakeConnection(fake_resp)
        client.get_connection = lambda *a, **kw: conn
        resp, body = client.json_request('GET', '/v1/resources')
        self.assertEqual(resp, fake_resp)
        self.assertIsInstance(body, list)

    def test_server_success_body_json(self):
        err = _get_error_body()
        fake_resp = utils.FakeResponse({'content-type': 'application/json'}, io.StringIO(err), version=1, status=200)
        client = http.HTTPClient('http://localhost/')
        conn = utils.FakeConnection(fake_resp)
        client.get_connection = lambda *a, **kw: conn
        resp, body = client.json_request('GET', '/v1/resources')
        self.assertEqual(resp, fake_resp)
        self.assertEqual(jsonutils.dumps(body), err)

    def test_raw_request(self):
        fake_resp = utils.FakeResponse({'content-type': 'application/octet-stream'}, 'bar', version=1, status=200)
        client = http.HTTPClient('http://localhost/')
        conn = utils.FakeConnection(fake_resp)
        client.get_connection = lambda *a, **kw: conn
        resp, body = client.raw_request('GET', '/v1/resources')
        self.assertEqual(resp, fake_resp)
        self.assertIsInstance(body, http.ResponseBodyIterator)