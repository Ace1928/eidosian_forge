import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
class TestAWSAuthConnection(unittest.TestCase):

    def test_get_path(self):
        conn = AWSAuthConnection('mockservice.cc-zone-1.amazonaws.com', aws_access_key_id='access_key', aws_secret_access_key='secret', suppress_consec_slashes=False)
        self.assertEqual(conn.get_path('/'), '/')
        self.assertEqual(conn.get_path('image.jpg'), '/image.jpg')
        self.assertEqual(conn.get_path('folder/image.jpg'), '/folder/image.jpg')
        self.assertEqual(conn.get_path('folder//image.jpg'), '/folder//image.jpg')
        self.assertEqual(conn.get_path('/folder//image.jpg'), '/folder//image.jpg')
        self.assertEqual(conn.get_path('/folder////image.jpg'), '/folder////image.jpg')
        self.assertEqual(conn.get_path('///folder////image.jpg'), '///folder////image.jpg')

    def test_connection_behind_proxy(self):
        os.environ['http_proxy'] = 'http://john.doe:p4ssw0rd@127.0.0.1:8180'
        conn = AWSAuthConnection('mockservice.cc-zone-1.amazonaws.com', aws_access_key_id='access_key', aws_secret_access_key='secret', suppress_consec_slashes=False)
        self.assertEqual(conn.proxy, '127.0.0.1')
        self.assertEqual(conn.proxy_user, 'john.doe')
        self.assertEqual(conn.proxy_pass, 'p4ssw0rd')
        self.assertEqual(conn.proxy_port, '8180')
        del os.environ['http_proxy']

    def test_get_proxy_url_with_auth(self):
        conn = AWSAuthConnection('mockservice.cc-zone-1.amazonaws.com', aws_access_key_id='access_key', aws_secret_access_key='secret', suppress_consec_slashes=False, proxy='127.0.0.1', proxy_user='john.doe', proxy_pass='p4ssw0rd', proxy_port='8180')
        self.assertEqual(conn.get_proxy_url_with_auth(), 'http://john.doe:p4ssw0rd@127.0.0.1:8180')

    def test_build_base_http_request_noproxy(self):
        os.environ['no_proxy'] = 'mockservice.cc-zone-1.amazonaws.com'
        conn = AWSAuthConnection('mockservice.cc-zone-1.amazonaws.com', aws_access_key_id='access_key', aws_secret_access_key='secret', suppress_consec_slashes=False, proxy='127.0.0.1', proxy_user='john.doe', proxy_pass='p4ssw0rd', proxy_port='8180')
        request = conn.build_base_http_request('GET', '/', None)
        del os.environ['no_proxy']
        self.assertEqual(request.path, '/')

    def test_connection_behind_proxy_without_explicit_port(self):
        os.environ['http_proxy'] = 'http://127.0.0.1'
        conn = AWSAuthConnection('mockservice.cc-zone-1.amazonaws.com', aws_access_key_id='access_key', aws_secret_access_key='secret', suppress_consec_slashes=False, port=8180)
        self.assertEqual(conn.proxy, '127.0.0.1')
        self.assertEqual(conn.proxy_port, 8180)
        del os.environ['http_proxy']

    @mock.patch.object(socket, 'create_connection')
    @mock.patch('boto.compat.http_client.HTTPResponse')
    @mock.patch('boto.connection.ssl', autospec=True)
    def test_proxy_ssl_with_verification(self, ssl_mock, http_response_mock, create_connection_mock):
        type(http_response_mock.return_value).status = mock.PropertyMock(return_value=200)
        conn = AWSAuthConnection('mockservice.s3.amazonaws.com', aws_access_key_id='access_key', aws_secret_access_key='secret', suppress_consec_slashes=False, proxy_port=80)
        conn.https_validate_certificates = True
        dummy_cert = {'subjectAltName': (('DNS', 's3.amazonaws.com'), ('DNS', '*.s3.amazonaws.com'))}
        mock_sock = mock.Mock()
        create_connection_mock.return_value = mock_sock
        mock_sslSock = mock.Mock()
        mock_sslSock.getpeercert.return_value = dummy_cert
        mock_context = mock.Mock()
        mock_context.wrap_socket.return_value = mock_sslSock
        ssl_mock.create_default_context.return_value = mock_context
        conn.proxy_ssl('mockservice.s3.amazonaws.com', 80)
        mock_sslSock.getpeercert.assert_called_once_with()
        mock_context.wrap_socket.assert_called_once_with(mock_sock, server_hostname='mockservice.s3.amazonaws.com')

    def test_host_header_with_nonstandard_port(self):
        conn = V4AuthConnection('testhost', aws_access_key_id='access_key', aws_secret_access_key='secret')
        request = conn.build_base_http_request(method='POST', path='/', auth_path=None, params=None, headers=None, data='', host=None)
        conn.set_host_header(request)
        self.assertEqual(request.headers['Host'], 'testhost')
        conn = V4AuthConnection('testhost', aws_access_key_id='access_key', aws_secret_access_key='secret', port=8773)
        request = conn.build_base_http_request(method='POST', path='/', auth_path=None, params=None, headers=None, data='', host=None)
        conn.set_host_header(request)
        self.assertEqual(request.headers['Host'], 'testhost:8773')