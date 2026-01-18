import os
import ssl
import sys
import socket
from unittest import mock
from unittest.mock import Mock, patch
import requests_mock
from requests.exceptions import ConnectTimeout
import libcloud.common.base
from libcloud.http import LibcloudConnection, SignedHTTPSAdapter, LibcloudBaseConnection
from libcloud.test import unittest, no_internet
from libcloud.utils.py3 import assertRaisesRegex
from libcloud.common.base import Response, Connection, CertificateConnection
from libcloud.utils.retry import RETRY_EXCEPTIONS, Retry, RetryForeverOnRateLimitError
from libcloud.common.exceptions import RateLimitReachedError
class BaseConnectionClassTestCase(unittest.TestCase):

    def setUp(self):
        self.orig_http_proxy = os.environ.pop('http_proxy', None)
        self.orig_https_proxy = os.environ.pop('https_proxy', None)

    def tearDown(self):
        if self.orig_http_proxy:
            os.environ['http_proxy'] = self.orig_http_proxy
        elif 'http_proxy' in os.environ:
            del os.environ['http_proxy']
        if self.orig_https_proxy:
            os.environ['https_proxy'] = self.orig_https_proxy
        elif 'https_proxy' in os.environ:
            del os.environ['https_proxy']
        libcloud.common.base.ALLOW_PATH_DOUBLE_SLASHES = False

    def test_parse_proxy_url(self):
        conn = LibcloudBaseConnection()
        proxy_url = 'http://127.0.0.1:3128'
        result = conn._parse_proxy_url(proxy_url=proxy_url)
        self.assertEqual(result[0], 'http')
        self.assertEqual(result[1], '127.0.0.1')
        self.assertEqual(result[2], 3128)
        self.assertIsNone(result[3])
        self.assertIsNone(result[4])
        proxy_url = 'https://127.0.0.2:3129'
        result = conn._parse_proxy_url(proxy_url=proxy_url)
        self.assertEqual(result[0], 'https')
        self.assertEqual(result[1], '127.0.0.2')
        self.assertEqual(result[2], 3129)
        self.assertIsNone(result[3])
        self.assertIsNone(result[4])
        proxy_url = 'http://user1:pass1@127.0.0.1:3128'
        result = conn._parse_proxy_url(proxy_url=proxy_url)
        self.assertEqual(result[0], 'http')
        self.assertEqual(result[1], '127.0.0.1')
        self.assertEqual(result[2], 3128)
        self.assertEqual(result[3], 'user1')
        self.assertEqual(result[4], 'pass1')
        proxy_url = 'https://user1:pass1@127.0.0.2:3129'
        result = conn._parse_proxy_url(proxy_url=proxy_url)
        self.assertEqual(result[0], 'https')
        self.assertEqual(result[1], '127.0.0.2')
        self.assertEqual(result[2], 3129)
        self.assertEqual(result[3], 'user1')
        self.assertEqual(result[4], 'pass1')
        proxy_url = 'http://127.0.0.1'
        expected_msg = 'proxy_url must be in the following format'
        assertRaisesRegex(self, ValueError, expected_msg, conn._parse_proxy_url, proxy_url=proxy_url)
        proxy_url = 'http://@127.0.0.1:3128'
        expected_msg = 'URL is in an invalid format'
        assertRaisesRegex(self, ValueError, expected_msg, conn._parse_proxy_url, proxy_url=proxy_url)
        proxy_url = 'http://user@127.0.0.1:3128'
        expected_msg = 'URL is in an invalid format'
        assertRaisesRegex(self, ValueError, expected_msg, conn._parse_proxy_url, proxy_url=proxy_url)

    def test_constructor(self):
        proxy_url = 'http://127.0.0.2:3128'
        os.environ['http_proxy'] = proxy_url
        conn = LibcloudConnection(host='localhost', port=80)
        self.assertEqual(conn.proxy_scheme, 'http')
        self.assertEqual(conn.proxy_host, '127.0.0.2')
        self.assertEqual(conn.proxy_port, 3128)
        self.assertEqual(conn.session.proxies, {'http': 'http://127.0.0.2:3128', 'https': 'http://127.0.0.2:3128'})
        _ = os.environ.pop('http_proxy', None)
        conn = LibcloudConnection(host='localhost', port=80)
        self.assertIsNone(conn.proxy_scheme)
        self.assertIsNone(conn.proxy_host)
        self.assertIsNone(conn.proxy_port)
        proxy_url = 'http://127.0.0.3:3128'
        conn.set_http_proxy(proxy_url=proxy_url)
        self.assertEqual(conn.proxy_scheme, 'http')
        self.assertEqual(conn.proxy_host, '127.0.0.3')
        self.assertEqual(conn.proxy_port, 3128)
        self.assertEqual(conn.session.proxies, {'http': 'http://127.0.0.3:3128', 'https': 'http://127.0.0.3:3128'})
        proxy_url = 'http://127.0.0.4:3128'
        conn = LibcloudConnection(host='localhost', port=80, proxy_url=proxy_url)
        self.assertEqual(conn.proxy_scheme, 'http')
        self.assertEqual(conn.proxy_host, '127.0.0.4')
        self.assertEqual(conn.proxy_port, 3128)
        self.assertEqual(conn.session.proxies, {'http': 'http://127.0.0.4:3128', 'https': 'http://127.0.0.4:3128'})
        os.environ['http_proxy'] = proxy_url
        proxy_url = 'http://127.0.0.5:3128'
        conn = LibcloudConnection(host='localhost', port=80, proxy_url=proxy_url)
        self.assertEqual(conn.proxy_scheme, 'http')
        self.assertEqual(conn.proxy_host, '127.0.0.5')
        self.assertEqual(conn.proxy_port, 3128)
        self.assertEqual(conn.session.proxies, {'http': 'http://127.0.0.5:3128', 'https': 'http://127.0.0.5:3128'})
        os.environ['http_proxy'] = proxy_url
        proxy_url = 'https://127.0.0.6:3129'
        conn = LibcloudConnection(host='localhost', port=80, proxy_url=proxy_url)
        self.assertEqual(conn.proxy_scheme, 'https')
        self.assertEqual(conn.proxy_host, '127.0.0.6')
        self.assertEqual(conn.proxy_port, 3129)
        self.assertEqual(conn.session.proxies, {'http': 'https://127.0.0.6:3129', 'https': 'https://127.0.0.6:3129'})

    def test_connection_to_unusual_port(self):
        conn = LibcloudConnection(host='localhost', port=8080)
        self.assertIsNone(conn.proxy_scheme)
        self.assertIsNone(conn.proxy_host)
        self.assertIsNone(conn.proxy_port)
        self.assertEqual(conn.host, 'http://localhost:8080')
        conn = LibcloudConnection(host='localhost', port=80)
        self.assertEqual(conn.host, 'http://localhost')

    def test_connection_session_timeout(self):
        """
        Test that the connection timeout attribute is set correctly
        """
        conn = LibcloudConnection(host='localhost', port=8080)
        self.assertEqual(conn.session.timeout, 60)
        conn = LibcloudConnection(host='localhost', port=8080, timeout=10)
        self.assertEqual(conn.session.timeout, 10)

    @unittest.skipIf(no_internet(), 'Internet is not reachable')
    def test_connection_timeout_raised(self):
        """
        Test that the connection times out
        """
        conn = LibcloudConnection(host='localhost', port=8080, timeout=0.1)
        host = 'http://10.255.255.1'
        with self.assertRaises(ConnectTimeout):
            conn.request('GET', host)

    def test_connection_url_merging(self):
        """
        Test that the connection class will parse URLs correctly
        """
        conn = Connection(url='http://test.com/')
        conn.connect()
        self.assertEqual(conn.connection.host, 'http://test.com')
        with requests_mock.mock() as m:
            m.get('http://test.com/test', text='data')
            response = conn.request('/test')
        self.assertEqual(response.body, 'data')

    def test_morph_action_hook(self):
        conn = Connection(url='http://test.com')
        conn.request_path = ''
        self.assertEqual(conn.morph_action_hook('/test'), '/test')
        self.assertEqual(conn.morph_action_hook('test'), '/test')
        conn.request_path = '/v1'
        self.assertEqual(conn.morph_action_hook('/test'), '/v1/test')
        self.assertEqual(conn.morph_action_hook('test'), '/v1/test')
        conn.request_path = '/v1'
        self.assertEqual(conn.morph_action_hook('/test'), '/v1/test')
        self.assertEqual(conn.morph_action_hook('test'), '/v1/test')
        conn.request_path = 'v1'
        self.assertEqual(conn.morph_action_hook('/test'), '/v1/test')
        self.assertEqual(conn.morph_action_hook('test'), '/v1/test')
        conn.request_path = 'v1/'
        self.assertEqual(conn.morph_action_hook('/test'), '/v1/test')
        self.assertEqual(conn.morph_action_hook('test'), '/v1/test')
        conn.request_path = '/a'
        self.assertEqual(conn.morph_action_hook('//b/c.txt'), '/a/b/c.txt')
        conn.request_path = '/b'
        self.assertEqual(conn.morph_action_hook('/foo//'), '/b/foo/')
        libcloud.common.base.ALLOW_PATH_DOUBLE_SLASHES = True
        conn.request_path = '/'
        self.assertEqual(conn.morph_action_hook('/'), '//')
        conn.request_path = ''
        self.assertEqual(conn.morph_action_hook('/'), '/')
        conn.request_path = '/a'
        self.assertEqual(conn.morph_action_hook('//b/c.txt'), '/a//b/c.txt')
        conn.request_path = '/b'
        self.assertEqual(conn.morph_action_hook('/foo//'), '/b/foo//')

    def test_connect_with_prefix(self):
        """
        Test that a connection with a base path (e.g. /v1/) will
        add the base path to requests
        """
        conn = Connection(url='http://test.com/')
        conn.connect()
        conn.request_path = '/v1'
        self.assertEqual(conn.connection.host, 'http://test.com')
        with requests_mock.mock() as m:
            m.get('http://test.com/v1/test', text='data')
            response = conn.request('/test')
        self.assertEqual(response.body, 'data')

    def test_secure_connection_unusual_port(self):
        """
        Test that the connection class will default to secure (https) even
        when the port is an unusual (non 443, 80) number
        """
        conn = Connection(secure=True, host='localhost', port=8081)
        conn.connect()
        self.assertEqual(conn.connection.host, 'https://localhost:8081')
        conn2 = Connection(url='https://localhost:8081')
        conn2.connect()
        self.assertEqual(conn2.connection.host, 'https://localhost:8081')

    def test_secure_by_default(self):
        """
        Test that the connection class will default to secure (https)
        """
        conn = Connection(host='localhost', port=8081)
        conn.connect()
        self.assertEqual(conn.connection.host, 'https://localhost:8081')

    def test_implicit_port(self):
        """
        Test that the port is not included in the URL if the protocol implies
        the port, e.g. http implies 80
        """
        conn = Connection(secure=True, host='localhost', port=443)
        conn.connect()
        self.assertEqual(conn.connection.host, 'https://localhost')
        conn2 = Connection(secure=False, host='localhost', port=80)
        conn2.connect()
        self.assertEqual(conn2.connection.host, 'http://localhost')

    def test_insecure_connection_unusual_port(self):
        """
        Test that the connection will allow unusual ports and insecure
        schemes
        """
        conn = Connection(secure=False, host='localhost', port=8081)
        conn.connect()
        self.assertEqual(conn.connection.host, 'http://localhost:8081')
        conn2 = Connection(url='http://localhost:8081')
        conn2.connect()
        self.assertEqual(conn2.connection.host, 'http://localhost:8081')