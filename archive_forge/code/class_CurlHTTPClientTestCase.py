from hashlib import md5
import unittest
from tornado.escape import utf8
from tornado.testing import AsyncHTTPTestCase
from tornado.test import httpclient_test
from tornado.web import Application, RequestHandler
@unittest.skipIf(pycurl is None, 'pycurl module not present')
class CurlHTTPClientTestCase(AsyncHTTPTestCase):

    def setUp(self):
        super().setUp()
        self.http_client = self.create_client()

    def get_app(self):
        return Application([('/digest', DigestAuthHandler, {'username': 'foo', 'password': 'bar'}), ('/digest_non_ascii', DigestAuthHandler, {'username': 'foo', 'password': 'barユ£'}), ('/custom_reason', CustomReasonHandler), ('/custom_fail_reason', CustomFailReasonHandler)])

    def create_client(self, **kwargs):
        return CurlAsyncHTTPClient(force_instance=True, defaults=dict(allow_ipv6=False), **kwargs)

    def test_digest_auth(self):
        response = self.fetch('/digest', auth_mode='digest', auth_username='foo', auth_password='bar')
        self.assertEqual(response.body, b'ok')

    def test_custom_reason(self):
        response = self.fetch('/custom_reason')
        self.assertEqual(response.reason, 'Custom reason')

    def test_fail_custom_reason(self):
        response = self.fetch('/custom_fail_reason')
        self.assertEqual(str(response.error), 'HTTP 400: Custom reason')

    def test_digest_auth_non_ascii(self):
        response = self.fetch('/digest_non_ascii', auth_mode='digest', auth_username='foo', auth_password='barユ£')
        self.assertEqual(response.body, b'ok')