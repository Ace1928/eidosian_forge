import os
import sys
import time
import random
import os.path
import platform
import warnings
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import libcloud.security
from libcloud.http import LibcloudConnection
from libcloud.test import unittest, no_network
from libcloud.utils.py3 import reload, httplib, assertRaisesRegex
@unittest.skipIf(platform.python_implementation() == 'PyPy', 'Skipping test under PyPy since it causes segfault')
class HttpLayerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.listen_host = '127.0.0.1'
        cls.listen_port = random.randint(10024, 65555)
        cls.mock_server = HTTPServer((cls.listen_host, cls.listen_port), MockHTTPServerRequestHandler)
        cls.mock_server_thread = threading.Thread(target=cls.mock_server.serve_forever)
        cls.mock_server_thread.setDaemon(True)
        cls.mock_server_thread.start()
        cls.orig_http_proxy = os.environ.pop('http_proxy', None)
        cls.orig_https_proxy = os.environ.pop('https_proxy', None)

    @classmethod
    def tearDownClass(cls):
        cls.mock_server.shutdown()
        cls.mock_server.server_close()
        cls.mock_server_thread.join()
        if cls.orig_http_proxy:
            os.environ['http_proxy'] = cls.orig_http_proxy
        elif 'http_proxy' in os.environ:
            del os.environ['http_proxy']
        if cls.orig_https_proxy:
            os.environ['https_proxy'] = cls.orig_https_proxy
        elif 'https_proxy' in os.environ:
            del os.environ['https_proxy']

    @unittest.skipIf(no_network(), 'Network is disabled')
    def test_prepared_request_empty_body_chunked_encoding_not_used(self):
        connection = LibcloudConnection(host=self.listen_host, port=self.listen_port)
        connection.prepared_request(method='GET', url='/test/prepared-request-1', body='', stream=True)
        self.assertEqual(connection.response.status_code, httplib.OK)
        self.assertEqual(connection.response.content, b'/test/prepared-request-1')
        connection = LibcloudConnection(host=self.listen_host, port=self.listen_port)
        connection.prepared_request(method='GET', url='/test/prepared-request-2', body=None, stream=True)
        self.assertEqual(connection.response.status_code, httplib.OK)
        self.assertEqual(connection.response.content, b'/test/prepared-request-2')

    @unittest.skipIf(no_network(), 'Network is disabled')
    def test_prepared_request_with_body(self):
        connection = LibcloudConnection(host=self.listen_host, port=self.listen_port)
        connection.prepared_request(method='GET', url='/test/prepared-request-3', body='test body', stream=True)
        self.assertEqual(connection.response.status_code, httplib.OK)
        self.assertEqual(connection.response.content, b'/test/prepared-request-3')

    @unittest.skipIf(no_network(), 'Network is disabled')
    def test_request_custom_timeout_no_timeout(self):

        def response_hook(*args, **kwargs):
            self.assertEqual(kwargs['timeout'], 5)
        hooks = {'response': response_hook}
        connection = LibcloudConnection(host=self.listen_host, port=self.listen_port, timeout=5)
        connection.request(method='GET', url='/test', hooks=hooks)

    @unittest.skipIf(no_network(), 'Network is disabled')
    def test_request_custom_timeout_timeout(self):

        def response_hook(*args, **kwargs):
            self.assertEqual(kwargs['timeout'], 0.5)
        hooks = {'response': response_hook}
        connection = LibcloudConnection(host=self.listen_host, port=self.listen_port, timeout=0.5)
        self.assertRaisesRegex(requests.exceptions.ReadTimeout, 'Read timed out', connection.request, method='GET', url='/test-timeout', hooks=hooks)