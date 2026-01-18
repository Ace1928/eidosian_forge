from unittest import mock
import fixtures
from urllib import parse as urlparse
import uuid
from testtools import matchers
from keystoneclient import session
from keystoneclient.tests.unit.v3 import client_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient import utils as client_utils
from keystoneclient.v3.contrib.oauth1 import access_tokens
from keystoneclient.v3.contrib.oauth1 import auth
from keystoneclient.v3.contrib.oauth1 import consumers
from keystoneclient.v3.contrib.oauth1 import request_tokens
class AuthenticateWithOAuthTests(utils.TestCase, TokenTests):

    def setUp(self):
        super(AuthenticateWithOAuthTests, self).setUp()
        if oauth1 is None:
            self.skipTest('optional package oauthlib is not installed')

    def test_oauth_authenticate_success(self):
        consumer_key = uuid.uuid4().hex
        consumer_secret = uuid.uuid4().hex
        access_key = uuid.uuid4().hex
        access_secret = uuid.uuid4().hex
        oauth_token = client_fixtures.project_scoped_token()
        oauth_token['methods'] = ['oauth1']
        oauth_token['OS-OAUTH1'] = {'consumer_id': consumer_key, 'access_token_id': access_key}
        self.stub_auth(json=oauth_token)
        with self.deprecations.expect_deprecations_here():
            a = auth.OAuth(self.TEST_URL, consumer_key=consumer_key, consumer_secret=consumer_secret, access_key=access_key, access_secret=access_secret)
            s = session.Session(auth=a)
            t = s.get_token()
        self.assertEqual(self.TEST_TOKEN, t)
        OAUTH_REQUEST_BODY = {'auth': {'identity': {'methods': ['oauth1'], 'oauth1': {}}}}
        self.assertRequestBodyIs(json=OAUTH_REQUEST_BODY)
        req_headers = self.requests_mock.last_request.headers
        oauth_client = oauth1.Client(consumer_key, client_secret=consumer_secret, resource_owner_key=access_key, resource_owner_secret=access_secret, signature_method=oauth1.SIGNATURE_HMAC)
        self._validate_oauth_headers(req_headers['Authorization'], oauth_client)