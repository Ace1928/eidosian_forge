import io
import logging
from testtools import matchers
from keystoneclient import exceptions
from keystoneclient import httpclient
from keystoneclient import session
from keystoneclient.tests.unit import utils
class BasicRequestTests(utils.TestCase):
    url = 'http://keystone.test.com/'

    def setUp(self):
        super(BasicRequestTests, self).setUp()
        self.logger_message = io.StringIO()
        handler = logging.StreamHandler(self.logger_message)
        handler.setLevel(logging.DEBUG)
        self.logger = logging.getLogger(session.__name__)
        level = self.logger.getEffectiveLevel()
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)
        self.addCleanup(self.logger.removeHandler, handler)
        self.addCleanup(self.logger.setLevel, level)

    def request(self, method='GET', response='Test Response', status_code=200, url=None, headers={}, **kwargs):
        if not url:
            url = self.url
        self.requests_mock.register_uri(method, url, text=response, status_code=status_code, headers=headers)
        with self.deprecations.expect_deprecations_here():
            return httpclient.request(url, method, headers=headers, **kwargs)

    def test_basic_params(self):
        method = 'GET'
        response = 'Test Response'
        status = 200
        self.request(method=method, status_code=status, response=response, headers={'Content-Type': 'application/json'})
        self.assertEqual(self.requests_mock.last_request.method, method)
        logger_message = self.logger_message.getvalue()
        self.assertThat(logger_message, matchers.Contains('curl'))
        self.assertThat(logger_message, matchers.Contains('-X %s' % method))
        self.assertThat(logger_message, matchers.Contains(self.url))
        self.assertThat(logger_message, matchers.Contains(str(status)))
        self.assertThat(logger_message, matchers.Contains(response))

    def test_headers(self):
        headers = {'key': 'val', 'test': 'other'}
        self.request(headers=headers)
        for k, v in headers.items():
            self.assertRequestHeaderEqual(k, v)
        for header in headers.items():
            self.assertThat(self.logger_message.getvalue(), matchers.Contains('-H "%s: %s"' % header))

    def test_body(self):
        data = 'BODY DATA'
        self.request(response=data, headers={'Content-Type': 'application/json'})
        logger_message = self.logger_message.getvalue()
        self.assertThat(logger_message, matchers.Contains('BODY:'))
        self.assertThat(logger_message, matchers.Contains(data))