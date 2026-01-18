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
class S3TokenMiddlewareTestBase(utils.TestCase):
    TEST_WWW_AUTHENTICATE_URI = 'https://fakehost/identity'
    TEST_URL = '%s/v2.0/s3tokens' % (TEST_WWW_AUTHENTICATE_URI,)

    def setUp(self):
        super(S3TokenMiddlewareTestBase, self).setUp()
        self.conf = {'www_authenticate_uri': self.TEST_WWW_AUTHENTICATE_URI}
        self.requests_mock = self.useFixture(rm_fixture.Fixture())

    def start_fake_response(self, status, headers):
        self.response_status = int(status.split(' ', 1)[0])
        self.response_headers = dict(headers)