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
class RequestTokenTests(utils.ClientTestCase, TokenTests):

    def setUp(self):
        if oauth1 is None:
            self.skipTest('oauthlib package not available')
        super(RequestTokenTests, self).setUp()
        self.model = request_tokens.RequestToken
        self.manager = self.client.oauth1.request_tokens
        self.path_prefix = 'OS-OAUTH1'

    def test_authorize_request_token(self):
        request_key = uuid.uuid4().hex
        info = {'id': request_key, 'key': request_key, 'secret': uuid.uuid4().hex}
        request_token = request_tokens.RequestToken(self.manager, info)
        verifier = uuid.uuid4().hex
        resp_ref = {'token': {'oauth_verifier': verifier}}
        self.stub_url('PUT', [self.path_prefix, 'authorize', request_key], status_code=200, json=resp_ref)
        role_id = uuid.uuid4().hex
        token = request_token.authorize([role_id])
        self.assertEqual(verifier, token.oauth_verifier)
        exp_body = {'roles': [{'id': role_id}]}
        self.assertRequestBodyIs(json=exp_body)

    def test_create_request_token(self):
        project_id = uuid.uuid4().hex
        consumer_key = uuid.uuid4().hex
        consumer_secret = uuid.uuid4().hex
        request_key, request_secret, resp_ref = self._new_oauth_token()
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        self.stub_url('POST', [self.path_prefix, 'request_token'], status_code=201, text=resp_ref, headers=headers)
        request_token = self.manager.create(consumer_key, consumer_secret, project_id)
        self.assertIsInstance(request_token, self.model)
        self.assertEqual(request_key, request_token.key)
        self.assertEqual(request_secret, request_token.secret)
        self.assertRequestHeaderEqual('requested-project-id', project_id)
        req_headers = self.requests_mock.last_request.headers
        oauth_client = oauth1.Client(consumer_key, client_secret=consumer_secret, signature_method=oauth1.SIGNATURE_HMAC, callback_uri='oob')
        self._validate_oauth_headers(req_headers['Authorization'], oauth_client)