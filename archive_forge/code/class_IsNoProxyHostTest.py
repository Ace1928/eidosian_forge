import os
import unittest
from websocket._url import (
class IsNoProxyHostTest(unittest.TestCase):

    def setUp(self):
        self.no_proxy = os.environ.get('no_proxy', None)
        if 'no_proxy' in os.environ:
            del os.environ['no_proxy']

    def tearDown(self):
        if self.no_proxy:
            os.environ['no_proxy'] = self.no_proxy
        elif 'no_proxy' in os.environ:
            del os.environ['no_proxy']

    def testMatchAll(self):
        self.assertTrue(_is_no_proxy_host('any.websocket.org', ['*']))
        self.assertTrue(_is_no_proxy_host('192.168.0.1', ['*']))
        self.assertTrue(_is_no_proxy_host('any.websocket.org', ['other.websocket.org', '*']))
        os.environ['no_proxy'] = '*'
        self.assertTrue(_is_no_proxy_host('any.websocket.org', None))
        self.assertTrue(_is_no_proxy_host('192.168.0.1', None))
        os.environ['no_proxy'] = 'other.websocket.org, *'
        self.assertTrue(_is_no_proxy_host('any.websocket.org', None))

    def testIpAddress(self):
        self.assertTrue(_is_no_proxy_host('127.0.0.1', ['127.0.0.1']))
        self.assertFalse(_is_no_proxy_host('127.0.0.2', ['127.0.0.1']))
        self.assertTrue(_is_no_proxy_host('127.0.0.1', ['other.websocket.org', '127.0.0.1']))
        self.assertFalse(_is_no_proxy_host('127.0.0.2', ['other.websocket.org', '127.0.0.1']))
        os.environ['no_proxy'] = '127.0.0.1'
        self.assertTrue(_is_no_proxy_host('127.0.0.1', None))
        self.assertFalse(_is_no_proxy_host('127.0.0.2', None))
        os.environ['no_proxy'] = 'other.websocket.org, 127.0.0.1'
        self.assertTrue(_is_no_proxy_host('127.0.0.1', None))
        self.assertFalse(_is_no_proxy_host('127.0.0.2', None))

    def testIpAddressInRange(self):
        self.assertTrue(_is_no_proxy_host('127.0.0.1', ['127.0.0.0/8']))
        self.assertTrue(_is_no_proxy_host('127.0.0.2', ['127.0.0.0/8']))
        self.assertFalse(_is_no_proxy_host('127.1.0.1', ['127.0.0.0/24']))
        os.environ['no_proxy'] = '127.0.0.0/8'
        self.assertTrue(_is_no_proxy_host('127.0.0.1', None))
        self.assertTrue(_is_no_proxy_host('127.0.0.2', None))
        os.environ['no_proxy'] = '127.0.0.0/24'
        self.assertFalse(_is_no_proxy_host('127.1.0.1', None))

    def testHostnameMatch(self):
        self.assertTrue(_is_no_proxy_host('my.websocket.org', ['my.websocket.org']))
        self.assertTrue(_is_no_proxy_host('my.websocket.org', ['other.websocket.org', 'my.websocket.org']))
        self.assertFalse(_is_no_proxy_host('my.websocket.org', ['other.websocket.org']))
        os.environ['no_proxy'] = 'my.websocket.org'
        self.assertTrue(_is_no_proxy_host('my.websocket.org', None))
        self.assertFalse(_is_no_proxy_host('other.websocket.org', None))
        os.environ['no_proxy'] = 'other.websocket.org, my.websocket.org'
        self.assertTrue(_is_no_proxy_host('my.websocket.org', None))

    def testHostnameMatchDomain(self):
        self.assertTrue(_is_no_proxy_host('any.websocket.org', ['.websocket.org']))
        self.assertTrue(_is_no_proxy_host('my.other.websocket.org', ['.websocket.org']))
        self.assertTrue(_is_no_proxy_host('any.websocket.org', ['my.websocket.org', '.websocket.org']))
        self.assertFalse(_is_no_proxy_host('any.websocket.com', ['.websocket.org']))
        os.environ['no_proxy'] = '.websocket.org'
        self.assertTrue(_is_no_proxy_host('any.websocket.org', None))
        self.assertTrue(_is_no_proxy_host('my.other.websocket.org', None))
        self.assertFalse(_is_no_proxy_host('any.websocket.com', None))
        os.environ['no_proxy'] = 'my.websocket.org, .websocket.org'
        self.assertTrue(_is_no_proxy_host('any.websocket.org', None))