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
class AccessTokenTests(utils.ClientTestCase, TokenTests):

    def setUp(self):
        if oauth1 is None:
            self.skipTest('oauthlib package not available')
        super(AccessTokenTests, self).setUp()
        self.manager = self.client.oauth1.access_tokens
        self.model = access_tokens.AccessToken
        self.path_prefix = 'OS-OAUTH1'

    def test_create_access_token_expires_at(self):
        verifier = uuid.uuid4().hex
        consumer_key = uuid.uuid4().hex
        consumer_secret = uuid.uuid4().hex
        request_key = uuid.uuid4().hex
        request_secret = uuid.uuid4().hex
        t = self._new_oauth_token_with_expires_at()
        access_key, access_secret, expires_at, resp_ref = t
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        self.stub_url('POST', [self.path_prefix, 'access_token'], status_code=201, text=resp_ref, headers=headers)
        access_token = self.manager.create(consumer_key, consumer_secret, request_key, request_secret, verifier)
        self.assertIsInstance(access_token, self.model)
        self.assertEqual(access_key, access_token.key)
        self.assertEqual(access_secret, access_token.secret)
        self.assertEqual(expires_at, access_token.expires)
        req_headers = self.requests_mock.last_request.headers
        oauth_client = oauth1.Client(consumer_key, client_secret=consumer_secret, resource_owner_key=request_key, resource_owner_secret=request_secret, signature_method=oauth1.SIGNATURE_HMAC, verifier=verifier)
        self._validate_oauth_headers(req_headers['Authorization'], oauth_client)