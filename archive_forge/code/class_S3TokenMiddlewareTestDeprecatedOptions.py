from unittest import mock
import urllib.parse
import fixtures
from oslo_serialization import jsonutils
import requests
from requests_mock.contrib import fixture as rm_fixture
from testtools import matchers
import webob
from keystonemiddleware import s3_token
from keystonemiddleware.tests.unit import utils
class S3TokenMiddlewareTestDeprecatedOptions(S3TokenMiddlewareTestBase):

    def setUp(self):
        super(S3TokenMiddlewareTestDeprecatedOptions, self).setUp()
        self.conf = {'auth_uri': self.TEST_WWW_AUTHENTICATE_URI}
        self.logger = self.useFixture(fixtures.FakeLogger())
        self.middleware = s3_token.S3Token(FakeApp(), self.conf)
        self.requests_mock.post(self.TEST_URL, status_code=201, json=GOOD_RESPONSE)

    def test_logs_warning(self):
        req = webob.Request.blank('/')
        self.middleware(req.environ, self.start_fake_response)
        self.assertEqual(self.response_status, 200)
        log = 'Use of the auth_uri option was deprecated in the Queens release in favor of www_authenticate_uri.'
        self.assertThat(self.logger.output, matchers.Contains(log))