import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
class ApplicationCredentialAuth(test_v3.RestfulTestCase):

    def setUp(self):
        super(ApplicationCredentialAuth, self).setUp()
        self.app_cred_api = PROVIDERS.application_credential_api

    def config_overrides(self):
        super(ApplicationCredentialAuth, self).config_overrides()
        self.auth_plugin_config_override(methods=['application_credential', 'password', 'token'])

    def _make_app_cred(self, expires=None, access_rules=None):
        roles = [{'id': self.role_id}]
        data = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex, 'secret': uuid.uuid4().hex, 'user_id': self.user['id'], 'project_id': self.project['id'], 'description': uuid.uuid4().hex, 'roles': roles}
        if expires:
            data['expires_at'] = expires
        if access_rules:
            data['access_rules'] = access_rules
        return data

    def _validate_token(self, token, headers=None, expected_status=http.client.OK):
        path = '/v3/auth/tokens'
        headers = headers or {}
        headers.update({'X-Auth-Token': token, 'X-Subject-Token': token})
        with self.test_client() as c:
            resp = c.get(path, headers=headers, expected_status_code=expected_status)
        return resp

    def test_valid_application_credential_succeeds(self):
        app_cred = self._make_app_cred()
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        auth_data = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'])
        self.v3_create_token(auth_data, expected_status=http.client.CREATED)

    def test_validate_application_credential_token_populates_restricted(self):
        self.config_fixture.config(group='token', cache_on_issue=False)
        app_cred = self._make_app_cred()
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        auth_data = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'])
        auth_response = self.v3_create_token(auth_data, expected_status=http.client.CREATED)
        self.assertTrue(auth_response.json['token']['application_credential']['restricted'])
        token_id = auth_response.headers.get('X-Subject-Token')
        headers = {'X-Auth-Token': token_id, 'X-Subject-Token': token_id}
        validate_response = self.get('/auth/tokens', headers=headers).json_body
        self.assertTrue(validate_response['token']['application_credential']['restricted'])

    def test_valid_application_credential_with_name_succeeds(self):
        app_cred = self._make_app_cred()
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        auth_data = self.build_authentication_request(app_cred_name=app_cred_ref['name'], secret=app_cred_ref['secret'], user_id=self.user['id'])
        self.v3_create_token(auth_data, expected_status=http.client.CREATED)

    def test_valid_application_credential_name_and_username_succeeds(self):
        app_cred = self._make_app_cred()
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        auth_data = self.build_authentication_request(app_cred_name=app_cred_ref['name'], secret=app_cred_ref['secret'], username=self.user['name'], user_domain_id=self.user['domain_id'])
        self.v3_create_token(auth_data, expected_status=http.client.CREATED)

    def test_application_credential_with_invalid_secret_fails(self):
        app_cred = self._make_app_cred()
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        auth_data = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret='badsecret')
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_unexpired_application_credential_succeeds(self):
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=1)
        app_cred = self._make_app_cred(expires=expires_at)
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        auth_data = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'])
        self.v3_create_token(auth_data, expected_status=http.client.CREATED)

    def test_expired_application_credential_fails(self):
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=1)
        app_cred = self._make_app_cred(expires=expires_at)
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        auth_data = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'])
        future = datetime.datetime.utcnow() + datetime.timedelta(minutes=2)
        with freezegun.freeze_time(future):
            self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_application_credential_expiration_limits_token_expiration(self):
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=1)
        app_cred = self._make_app_cred(expires=expires_at)
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        auth_data = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'])
        resp = self.v3_create_token(auth_data, expected_status=http.client.CREATED)
        token = resp.headers.get('X-Subject-Token')
        future = datetime.datetime.utcnow() + datetime.timedelta(minutes=2)
        with freezegun.freeze_time(future):
            self._validate_token(token, expected_status=http.client.UNAUTHORIZED)

    def test_application_credential_fails_when_user_deleted(self):
        app_cred = self._make_app_cred()
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        PROVIDERS.identity_api.delete_user(self.user['id'])
        auth_data = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'])
        self.v3_create_token(auth_data, expected_status=http.client.NOT_FOUND)

    def test_application_credential_fails_when_user_disabled(self):
        app_cred = self._make_app_cred()
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        PROVIDERS.identity_api.update_user(self.user['id'], {'enabled': False})
        auth_data = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'])
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_application_credential_fails_when_project_deleted(self):
        app_cred = self._make_app_cred()
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        PROVIDERS.resource_api.delete_project(self.project['id'])
        auth_data = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'])
        self.v3_create_token(auth_data, expected_status=http.client.NOT_FOUND)

    def test_application_credential_fails_when_role_deleted(self):
        app_cred = self._make_app_cred()
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        PROVIDERS.role_api.delete_role(self.role_id)
        auth_data = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'])
        self.v3_create_token(auth_data, expected_status=http.client.NOT_FOUND)

    def test_application_credential_fails_when_role_unassigned(self):
        app_cred = self._make_app_cred()
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        PROVIDERS.assignment_api.remove_role_from_user_and_project(self.user['id'], self.project['id'], self.role_id)
        auth_data = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'])
        self.v3_create_token(auth_data, expected_status=http.client.NOT_FOUND)

    def test_application_credential_through_group_membership(self):
        user1 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
        group1 = unit.new_group_ref(domain_id=self.domain_id)
        group1 = PROVIDERS.identity_api.create_group(group1)
        PROVIDERS.identity_api.add_user_to_group(user1['id'], group1['id'])
        PROVIDERS.assignment_api.create_grant(self.role_id, group_id=group1['id'], project_id=self.project_id)
        app_cred = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex, 'secret': uuid.uuid4().hex, 'user_id': user1['id'], 'project_id': self.project_id, 'description': uuid.uuid4().hex, 'roles': [{'id': self.role_id}]}
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        auth_data = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'])
        self.v3_create_token(auth_data, expected_status=http.client.CREATED)

    def test_application_credential_cannot_scope(self):
        app_cred = self._make_app_cred()
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        new_project_ref = unit.new_project_ref(domain_id=self.domain_id)
        new_project = PROVIDERS.resource_api.create_project(new_project_ref['id'], new_project_ref)
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.user['id'], new_project['id'], self.role_id)
        password_auth = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=new_project['id'])
        password_response = self.v3_create_token(password_auth)
        self.assertValidProjectScopedTokenResponse(password_response)
        app_cred_auth = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'], project_id=new_project['id'])
        self.v3_create_token(app_cred_auth, expected_status=http.client.UNAUTHORIZED)

    def test_application_credential_with_access_rules(self):
        access_rules = [{'id': uuid.uuid4().hex, 'path': '/v2.1/servers', 'method': 'POST', 'service': uuid.uuid4().hex}]
        app_cred = self._make_app_cred(access_rules=access_rules)
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        auth_data = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'])
        resp = self.v3_create_token(auth_data, expected_status=http.client.CREATED)
        token = resp.headers.get('X-Subject-Token')
        headers = {'OpenStack-Identity-Access-Rules': '1.0'}
        self._validate_token(token, headers=headers)

    def test_application_credential_access_rules_without_header_fails(self):
        access_rules = [{'id': uuid.uuid4().hex, 'path': '/v2.1/servers', 'method': 'POST', 'service': uuid.uuid4().hex}]
        app_cred = self._make_app_cred(access_rules=access_rules)
        app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
        auth_data = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'])
        resp = self.v3_create_token(auth_data, expected_status=http.client.CREATED)
        token = resp.headers.get('X-Subject-Token')
        self._validate_token(token, expected_status=http.client.NOT_FOUND)