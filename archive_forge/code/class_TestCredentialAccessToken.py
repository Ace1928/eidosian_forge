import hashlib
import json
from unittest import mock
import uuid
import http.client
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_db import exception as oslo_db_exception
from testtools import matchers
import urllib
from keystone.api import ec2tokens
from keystone.common import provider_api
from keystone.common import utils
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone import oauth1
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
class TestCredentialAccessToken(CredentialBaseTestCase):
    """Test credential with access token."""

    def setUp(self):
        super(TestCredentialAccessToken, self).setUp()
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'credential', credential_fernet.MAX_ACTIVE_KEYS))
        self.base_url = 'http://localhost/v3'

    def _urllib_parse_qs_text_keys(self, content):
        results = urllib.parse.parse_qs(content)
        return {key.decode('utf-8'): value for key, value in results.items()}

    def _create_single_consumer(self):
        endpoint = '/OS-OAUTH1/consumers'
        ref = {'description': uuid.uuid4().hex}
        resp = self.post(endpoint, body={'consumer': ref})
        return resp.result['consumer']

    def _create_request_token(self, consumer, project_id, base_url=None):
        endpoint = '/OS-OAUTH1/request_token'
        client = oauth1.Client(consumer['key'], client_secret=consumer['secret'], signature_method=oauth1.SIG_HMAC, callback_uri='oob')
        headers = {'requested_project_id': project_id}
        if not base_url:
            base_url = self.base_url
        url, headers, body = client.sign(base_url + endpoint, http_method='POST', headers=headers)
        return (endpoint, headers)

    def _create_access_token(self, consumer, token, base_url=None):
        endpoint = '/OS-OAUTH1/access_token'
        client = oauth1.Client(consumer['key'], client_secret=consumer['secret'], resource_owner_key=token.key, resource_owner_secret=token.secret, signature_method=oauth1.SIG_HMAC, verifier=token.verifier)
        if not base_url:
            base_url = self.base_url
        url, headers, body = client.sign(base_url + endpoint, http_method='POST')
        headers.update({'Content-Type': 'application/json'})
        return (endpoint, headers)

    def _get_oauth_token(self, consumer, token):
        client = oauth1.Client(consumer['key'], client_secret=consumer['secret'], resource_owner_key=token.key, resource_owner_secret=token.secret, signature_method=oauth1.SIG_HMAC)
        endpoint = '/auth/tokens'
        url, headers, body = client.sign(self.base_url + endpoint, http_method='POST')
        headers.update({'Content-Type': 'application/json'})
        ref = {'auth': {'identity': {'oauth1': {}, 'methods': ['oauth1']}}}
        return (endpoint, headers, ref)

    def _authorize_request_token(self, request_id):
        if isinstance(request_id, bytes):
            request_id = request_id.decode()
        return '/OS-OAUTH1/authorize/%s' % request_id

    def _get_access_token(self):
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        consumer = {'key': consumer_id, 'secret': consumer_secret}
        url, headers = self._create_request_token(consumer, self.project_id)
        content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        credentials = self._urllib_parse_qs_text_keys(content.result)
        request_key = credentials['oauth_token'][0]
        request_secret = credentials['oauth_token_secret'][0]
        request_token = oauth1.Token(request_key, request_secret)
        url = self._authorize_request_token(request_key)
        body = {'roles': [{'id': self.role_id}]}
        resp = self.put(url, body=body, expected_status=http.client.OK)
        verifier = resp.result['token']['oauth_verifier']
        request_token.set_verifier(verifier)
        url, headers = self._create_access_token(consumer, request_token)
        content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        credentials = self._urllib_parse_qs_text_keys(content.result)
        access_key = credentials['oauth_token'][0]
        access_secret = credentials['oauth_token_secret'][0]
        access_token = oauth1.Token(access_key, access_secret)
        url, headers, body = self._get_oauth_token(consumer, access_token)
        content = self.post(url, headers=headers, body=body)
        return (access_key, content.headers['X-Subject-Token'])

    def test_access_token_ec2_credential(self):
        """Test creating ec2 credential from an oauth access token.

        Call ``POST /credentials``.
        """
        access_key, token_id = self._get_access_token()
        blob, ref = unit.new_ec2_credential(user_id=self.user_id, project_id=self.project_id)
        r = self.post('/credentials', body={'credential': ref}, token=token_id)
        ret_ref = ref.copy()
        ret_blob = blob.copy()
        ret_blob['access_token_id'] = access_key.decode('utf-8')
        ret_ref['blob'] = json.dumps(ret_blob)
        self.assertValidCredentialResponse(r, ref=ret_ref)
        access = blob['access'].encode('utf-8')
        self.assertEqual(hashlib.sha256(access).hexdigest(), r.result['credential']['id'])
        role = unit.new_role_ref(name='reader')
        role_id = role['id']
        PROVIDERS.role_api.create_role(role_id, role)
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_id, self.project_id, role_id)
        ret_blob = json.loads(r.result['credential']['blob'])
        ec2token = self._test_get_token(access=ret_blob['access'], secret=ret_blob['secret'])
        ec2_roles = [role['id'] for role in ec2token['roles']]
        self.assertIn(self.role_id, ec2_roles)
        self.assertNotIn(role_id, ec2_roles)