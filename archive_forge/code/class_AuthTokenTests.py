import copy
import datetime
import random
from unittest import mock
import uuid
import freezegun
import http.client
from oslo_serialization import jsonutils
from pycadf import cadftaxonomy
import urllib
from urllib import parse as urlparse
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import oauth1
from keystone.oauth1.backends import base
from keystone.tests import unit
from keystone.tests.unit.common import test_notifications
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
class AuthTokenTests(object):

    def test_keystone_token_is_valid(self):
        self.test_oauth_flow()
        headers = {'X-Subject-Token': self.keystone_token_id, 'X-Auth-Token': self.keystone_token_id}
        r = self.get('/auth/tokens', headers=headers)
        self.assertValidTokenResponse(r, self.user)
        oauth_section = r.result['token']['OS-OAUTH1']
        self.assertEqual(self.access_token.key.decode(), oauth_section['access_token_id'])
        self.assertEqual(self.consumer['key'], oauth_section['consumer_id'])
        roles_list = r.result['token']['roles']
        self.assertEqual(self.role_id, roles_list[0]['id'])
        ref = unit.new_user_ref(domain_id=self.domain_id)
        r = self.admin_request(path='/v3/users', headers=headers, method='POST', body={'user': ref})
        self.assertValidUserResponse(r, ref)

    def test_delete_access_token_also_revokes_token(self):
        self.test_oauth_flow()
        access_token_key = self.access_token.key.decode()
        resp = self.delete('/users/%(user)s/OS-OAUTH1/access_tokens/%(auth)s' % {'user': self.user_id, 'auth': access_token_key})
        self.assertResponseStatus(resp, http.client.NO_CONTENT)
        headers = {'X-Subject-Token': self.keystone_token_id, 'X-Auth-Token': self.keystone_token_id}
        self.get('/auth/tokens', headers=headers, expected_status=http.client.NOT_FOUND)

    def test_deleting_consumer_also_deletes_tokens(self):
        self.test_oauth_flow()
        consumer_id = self.consumer['key']
        resp = self.delete('/OS-OAUTH1/consumers/%(consumer_id)s' % {'consumer_id': consumer_id})
        self.assertResponseStatus(resp, http.client.NO_CONTENT)
        resp = self.get('/users/%(user_id)s/OS-OAUTH1/access_tokens' % {'user_id': self.user_id})
        entities = resp.result['access_tokens']
        self.assertEqual([], entities)
        headers = {'X-Subject-Token': self.keystone_token_id, 'X-Auth-Token': self.keystone_token_id}
        self.head('/auth/tokens', headers=headers, expected_status=http.client.NOT_FOUND)

    def test_change_user_password_also_deletes_tokens(self):
        self.test_oauth_flow()
        headers = {'X-Subject-Token': self.keystone_token_id, 'X-Auth-Token': self.keystone_token_id}
        r = self.get('/auth/tokens', headers=headers)
        self.assertValidTokenResponse(r, self.user)
        user = {'password': uuid.uuid4().hex}
        r = self.patch('/users/%(user_id)s' % {'user_id': self.user['id']}, body={'user': user})
        headers = {'X-Subject-Token': self.keystone_token_id}
        self.get(path='/auth/tokens', token=self.get_admin_token(), headers=headers, expected_status=http.client.NOT_FOUND)

    def test_deleting_project_also_invalidates_tokens(self):
        self.test_oauth_flow()
        headers = {'X-Subject-Token': self.keystone_token_id, 'X-Auth-Token': self.keystone_token_id}
        r = self.get('/auth/tokens', headers=headers)
        self.assertValidTokenResponse(r, self.user)
        r = self.delete('/projects/%(project_id)s' % {'project_id': self.project_id})
        headers = {'X-Subject-Token': self.keystone_token_id}
        self.get(path='/auth/tokens', token=self.get_admin_token(), headers=headers, expected_status=http.client.NOT_FOUND)

    def test_token_chaining_is_not_allowed(self):
        self.test_oauth_flow()
        path = '/v3/auth/tokens/'
        auth_data = self.build_authentication_request(token=self.keystone_token_id)
        self.admin_request(path=path, body=auth_data, token=self.keystone_token_id, method='POST', expected_status=http.client.FORBIDDEN)

    def test_delete_keystone_tokens_by_consumer_id(self):
        self.test_oauth_flow()
        PROVIDERS.token_provider_api._persistence.get_token(self.keystone_token_id)
        PROVIDERS.token_provider_api._persistence.delete_tokens(self.user_id, consumer_id=self.consumer['key'])
        self.assertRaises(exception.TokenNotFound, PROVIDERS.token_provider_api._persistence.get_token, self.keystone_token_id)

    def _create_trust_get_token(self):
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.user_id, project_id=self.project_id, impersonation=True, expires=dict(minutes=1), role_ids=[self.role_id])
        del ref['id']
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r)
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], trust_id=trust['id'])
        return self.get_requested_token(auth_data)

    def _approve_request_token_url(self):
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        self.consumer = {'key': consumer_id, 'secret': consumer_secret}
        self.assertIsNotNone(self.consumer['secret'])
        url, headers = self._create_request_token(self.consumer, self.project_id)
        content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        credentials = _urllib_parse_qs_text_keys(content.result)
        request_key = credentials['oauth_token'][0]
        request_secret = credentials['oauth_token_secret'][0]
        self.request_token = oauth1.Token(request_key, request_secret)
        self.assertIsNotNone(self.request_token.key)
        url = self._authorize_request_token(request_key)
        return url

    def test_oauth_token_cannot_create_new_trust(self):
        self.test_oauth_flow()
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.user_id, project_id=self.project_id, impersonation=True, expires=dict(minutes=1), role_ids=[self.role_id])
        del ref['id']
        self.post('/OS-TRUST/trusts', body={'trust': ref}, token=self.keystone_token_id, expected_status=http.client.FORBIDDEN)

    def test_oauth_token_cannot_authorize_request_token(self):
        self.test_oauth_flow()
        url = self._approve_request_token_url()
        body = {'roles': [{'id': self.role_id}]}
        self.put(url, body=body, token=self.keystone_token_id, expected_status=http.client.FORBIDDEN)

    def test_oauth_token_cannot_list_request_tokens(self):
        self._set_policy({'identity:list_access_tokens': [], 'identity:create_consumer': [], 'identity:authorize_request_token': []})
        self.test_oauth_flow()
        url = '/users/%s/OS-OAUTH1/access_tokens' % self.user_id
        self.get(url, token=self.keystone_token_id, expected_status=http.client.FORBIDDEN)

    def _set_policy(self, new_policy):
        self.tempfile = self.useFixture(temporaryfile.SecureTempFile())
        self.tmpfilename = self.tempfile.file_name
        self.config_fixture.config(group='oslo_policy', policy_file=self.tmpfilename)
        with open(self.tmpfilename, 'w') as policyfile:
            policyfile.write(jsonutils.dumps(new_policy))

    def test_trust_token_cannot_authorize_request_token(self):
        trust_token = self._create_trust_get_token()
        url = self._approve_request_token_url()
        body = {'roles': [{'id': self.role_id}]}
        self.put(url, body=body, token=trust_token, expected_status=http.client.FORBIDDEN)

    def test_trust_token_cannot_list_request_tokens(self):
        self._set_policy({'identity:list_access_tokens': [], 'identity:create_trust': []})
        trust_token = self._create_trust_get_token()
        url = '/users/%s/OS-OAUTH1/access_tokens' % self.user_id
        self.get(url, token=trust_token, expected_status=http.client.FORBIDDEN)