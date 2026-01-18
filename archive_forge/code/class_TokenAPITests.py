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
class TokenAPITests(object):

    def doSetUp(self):
        r = self.v3_create_token(self.build_authentication_request(username=self.user['name'], user_domain_id=self.domain_id, password=self.user['password']))
        self.v3_token_data = r.result
        self.v3_token = r.headers.get('X-Subject-Token')
        self.headers = {'X-Subject-Token': r.headers.get('X-Subject-Token')}

    def _get_unscoped_token(self):
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'])
        r = self.post('/auth/tokens', body=auth_data)
        self.assertValidUnscopedTokenResponse(r)
        return r.headers.get('X-Subject-Token')

    def _get_domain_scoped_token(self):
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_id=self.domain_id)
        r = self.post('/auth/tokens', body=auth_data)
        self.assertValidDomainScopedTokenResponse(r)
        return r.headers.get('X-Subject-Token')

    def _get_project_scoped_token(self):
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project_id)
        r = self.post('/auth/tokens', body=auth_data)
        self.assertValidProjectScopedTokenResponse(r)
        return r.headers.get('X-Subject-Token')

    def _get_trust_scoped_token(self, trustee_user, trust):
        auth_data = self.build_authentication_request(user_id=trustee_user['id'], password=trustee_user['password'], trust_id=trust['id'])
        r = self.post('/auth/tokens', body=auth_data)
        self.assertValidProjectScopedTokenResponse(r)
        return r.headers.get('X-Subject-Token')

    def _create_trust(self, impersonation=False):
        trustee_user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=trustee_user['id'], project_id=self.project_id, impersonation=impersonation, role_ids=[self.role_id])
        r = self.post('/OS-TRUST/trusts', body={'trust': ref})
        trust = self.assertValidTrustResponse(r)
        return (trustee_user, trust)

    def _validate_token(self, token, expected_status=http.client.OK, allow_expired=False):
        path = '/v3/auth/tokens'
        if allow_expired:
            path += '?allow_expired=1'
        return self.admin_request(path=path, headers={'X-Auth-Token': self.get_admin_token(), 'X-Subject-Token': token}, method='GET', expected_status=expected_status)

    def _revoke_token(self, token, expected_status=http.client.NO_CONTENT):
        return self.delete('/auth/tokens', headers={'x-subject-token': token}, expected_status=expected_status)

    def _set_user_enabled(self, user, enabled=True):
        user['enabled'] = enabled
        PROVIDERS.identity_api.update_user(user['id'], user)

    def _create_project_and_set_as_default_project(self):
        ref = unit.new_project_ref(domain_id=self.domain_id)
        r = self.post('/projects', body={'project': ref})
        project = self.assertValidProjectResponse(r, ref)
        self.put('/projects/%(project_id)s/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'project_id': project['id'], 'role_id': self.role['id']})
        body = {'user': {'default_project_id': project['id']}}
        r = self.patch('/users/%(user_id)s' % {'user_id': self.user['id']}, body=body)
        self.assertValidUserResponse(r)
        return project

    def test_auth_with_token_as_different_user_fails(self):
        token = self.get_scoped_token()
        auth_data = self.build_authentication_request(token=token, user_id=self.default_domain_user['id'], password=self.default_domain_user['password'])
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_create_token_for_user_without_password_fails(self):
        user = unit.new_user_ref(domain_id=self.domain['id'])
        del user['password']
        user = PROVIDERS.identity_api.create_user(user)
        auth_data = self.build_authentication_request(user_id=user['id'], password='password')
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_create_unscoped_token_by_authenticating_with_unscoped_token(self):
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'])
        r = self.v3_create_token(auth_data)
        self.assertValidUnscopedTokenResponse(r)
        token_id = r.headers.get('X-Subject-Token')
        auth_data = self.build_authentication_request(token=token_id)
        r = self.v3_create_token(auth_data)
        self.assertValidUnscopedTokenResponse(r)

    def test_create_unscoped_token_with_user_id(self):
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'])
        r = self.v3_create_token(auth_data)
        self.assertValidUnscopedTokenResponse(r)

    def test_create_unscoped_token_with_user_domain_id(self):
        auth_data = self.build_authentication_request(username=self.user['name'], user_domain_id=self.domain['id'], password=self.user['password'])
        r = self.v3_create_token(auth_data)
        self.assertValidUnscopedTokenResponse(r)

    def test_create_unscoped_token_with_user_domain_name(self):
        auth_data = self.build_authentication_request(username=self.user['name'], user_domain_name=self.domain['name'], password=self.user['password'])
        r = self.v3_create_token(auth_data)
        self.assertValidUnscopedTokenResponse(r)

    def test_validate_unscoped_token(self):
        unscoped_token = self._get_unscoped_token()
        r = self._validate_token(unscoped_token)
        self.assertValidUnscopedTokenResponse(r)

    def test_validate_expired_unscoped_token_returns_not_found(self):
        self.config_fixture.config(group='token', expiration=10)
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            unscoped_token = self._get_unscoped_token()
            frozen_datetime.tick(delta=datetime.timedelta(seconds=15))
            self._validate_token(unscoped_token, expected_status=http.client.NOT_FOUND)

    def test_revoke_unscoped_token(self):
        unscoped_token = self._get_unscoped_token()
        r = self._validate_token(unscoped_token)
        self.assertValidUnscopedTokenResponse(r)
        self._revoke_token(unscoped_token)
        self._validate_token(unscoped_token, expected_status=http.client.NOT_FOUND)

    def test_create_explicit_unscoped_token(self):
        self._create_project_and_set_as_default_project()
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], unscoped='unscoped')
        r = self.post('/auth/tokens', body=auth_data, noauth=True)
        self.assertValidUnscopedTokenResponse(r)

    def test_disabled_users_default_project_result_in_unscoped_token(self):
        project = self.create_new_default_project_for_user(self.user['id'], self.domain_id, enable_project=False)
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.user['id'], project['id'], self.role_id)
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'])
        r = self.v3_create_token(auth_data)
        self.assertValidUnscopedTokenResponse(r)

    def test_disabled_default_project_domain_result_in_unscoped_token(self):
        domain_ref = unit.new_domain_ref()
        r = self.post('/domains', body={'domain': domain_ref})
        domain = self.assertValidDomainResponse(r, domain_ref)
        project = self.create_new_default_project_for_user(self.user['id'], domain['id'])
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.user['id'], project['id'], self.role_id)
        body = {'domain': {'enabled': False}}
        r = self.patch('/domains/%(domain_id)s' % {'domain_id': domain['id']}, body=body)
        self.assertValidDomainResponse(r)
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'])
        r = self.v3_create_token(auth_data)
        self.assertValidUnscopedTokenResponse(r)

    def test_unscoped_token_is_invalid_after_disabling_user(self):
        unscoped_token = self._get_unscoped_token()
        r = self._validate_token(unscoped_token)
        self.assertValidUnscopedTokenResponse(r)
        self._set_user_enabled(self.user, enabled=False)
        self._validate_token(unscoped_token, expected_status=http.client.NOT_FOUND)

    def test_unscoped_token_is_invalid_after_enabling_disabled_user(self):
        unscoped_token = self._get_unscoped_token()
        r = self._validate_token(unscoped_token)
        self.assertValidUnscopedTokenResponse(r)
        self._set_user_enabled(self.user, enabled=False)
        self._validate_token(unscoped_token, expected_status=http.client.NOT_FOUND)
        self._set_user_enabled(self.user)
        self._validate_token(unscoped_token, expected_status=http.client.NOT_FOUND)

    def test_unscoped_token_is_invalid_after_disabling_user_domain(self):
        unscoped_token = self._get_unscoped_token()
        r = self._validate_token(unscoped_token)
        self.assertValidUnscopedTokenResponse(r)
        self.domain['enabled'] = False
        PROVIDERS.resource_api.update_domain(self.domain['id'], self.domain)
        self._validate_token(unscoped_token, expected_status=http.client.NOT_FOUND)

    def test_unscoped_token_is_invalid_after_changing_user_password(self):
        unscoped_token = self._get_unscoped_token()
        r = self._validate_token(unscoped_token)
        self.assertValidUnscopedTokenResponse(r)
        self.user['password'] = 'Password1'
        PROVIDERS.identity_api.update_user(self.user['id'], self.user)
        self._validate_token(unscoped_token, expected_status=http.client.NOT_FOUND)

    def test_create_system_token_with_user_id(self):
        path = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': self.role_id}
        self.put(path=path)
        auth_request_body = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], system=True)
        response = self.v3_create_token(auth_request_body)
        self.assertValidSystemScopedTokenResponse(response)

    def test_create_system_token_with_username(self):
        path = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': self.role_id}
        self.put(path=path)
        auth_request_body = self.build_authentication_request(username=self.user['name'], password=self.user['password'], user_domain_id=self.domain['id'], system=True)
        response = self.v3_create_token(auth_request_body)
        self.assertValidSystemScopedTokenResponse(response)

    def test_create_system_token_fails_without_system_assignment(self):
        auth_request_body = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], system=True)
        self.v3_create_token(auth_request_body, expected_status=http.client.UNAUTHORIZED)

    def test_system_token_is_invalid_after_disabling_user(self):
        path = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': self.role_id}
        self.put(path=path)
        auth_request_body = self.build_authentication_request(username=self.user['name'], password=self.user['password'], user_domain_id=self.domain['id'], system=True)
        response = self.v3_create_token(auth_request_body)
        self.assertValidSystemScopedTokenResponse(response)
        token = response.headers.get('X-Subject-Token')
        self._validate_token(token)
        user_ref = {'user': {'enabled': False}}
        self.patch('/users/%(user_id)s' % {'user_id': self.user['id']}, body=user_ref)
        self.admin_request(path='/v3/auth/tokens', headers={'X-Auth-Token': token, 'X-Subject-Token': token}, method='GET', expected_status=http.client.UNAUTHORIZED)
        self.admin_request(path='/v3/auth/tokens', headers={'X-Auth-Token': token, 'X-Subject-Token': token}, method='HEAD', expected_status=http.client.UNAUTHORIZED)

    def test_create_system_token_via_system_group_assignment(self):
        ref = {'group': unit.new_group_ref(domain_id=CONF.identity.default_domain_id)}
        group = self.post('/groups', body=ref).json_body['group']
        path = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': self.role_id}
        self.put(path=path)
        path = '/groups/%(group_id)s/users/%(user_id)s' % {'group_id': group['id'], 'user_id': self.user['id']}
        self.put(path=path)
        auth_request_body = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], system=True)
        response = self.v3_create_token(auth_request_body)
        self.assertValidSystemScopedTokenResponse(response)
        token = response.headers.get('X-Subject-Token')
        self._validate_token(token)

    def test_revoke_system_token(self):
        path = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': self.role_id}
        self.put(path=path)
        auth_request_body = self.build_authentication_request(username=self.user['name'], password=self.user['password'], user_domain_id=self.domain['id'], system=True)
        response = self.v3_create_token(auth_request_body)
        self.assertValidSystemScopedTokenResponse(response)
        token = response.headers.get('X-Subject-Token')
        self._validate_token(token)
        self._revoke_token(token)
        self._validate_token(token, expected_status=http.client.NOT_FOUND)

    def test_system_token_is_invalid_after_deleting_system_role(self):
        ref = {'role': unit.new_role_ref()}
        system_role = self.post('/roles', body=ref).json_body['role']
        path = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role['id']}
        self.put(path=path)
        auth_request_body = self.build_authentication_request(username=self.user['name'], password=self.user['password'], user_domain_id=self.domain['id'], system=True)
        response = self.v3_create_token(auth_request_body)
        self.assertValidSystemScopedTokenResponse(response)
        token = response.headers.get('X-Subject-Token')
        self._validate_token(token)
        self.delete('/roles/%(role_id)s' % {'role_id': system_role['id']})
        self._validate_token(token, expected_status=http.client.NOT_FOUND)

    def test_rescoping_a_system_token_for_a_project_token_fails(self):
        ref = {'role': unit.new_role_ref()}
        system_role = self.post('/roles', body=ref).json_body['role']
        path = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role['id']}
        self.put(path=path)
        auth_request_body = self.build_authentication_request(username=self.user['name'], password=self.user['password'], user_domain_id=self.domain['id'], system=True)
        response = self.v3_create_token(auth_request_body)
        self.assertValidSystemScopedTokenResponse(response)
        system_token = response.headers.get('X-Subject-Token')
        auth_request_body = self.build_authentication_request(token=system_token, project_id=self.project_id)
        self.v3_create_token(auth_request_body, expected_status=http.client.FORBIDDEN)

    def test_rescoping_a_system_token_for_a_domain_token_fails(self):
        ref = {'role': unit.new_role_ref()}
        system_role = self.post('/roles', body=ref).json_body['role']
        path = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role['id']}
        self.put(path=path)
        auth_request_body = self.build_authentication_request(username=self.user['name'], password=self.user['password'], user_domain_id=self.domain['id'], system=True)
        response = self.v3_create_token(auth_request_body)
        self.assertValidSystemScopedTokenResponse(response)
        system_token = response.headers.get('X-Subject-Token')
        auth_request_body = self.build_authentication_request(token=system_token, domain_id=CONF.identity.default_domain_id)
        self.v3_create_token(auth_request_body, expected_status=http.client.FORBIDDEN)

    def test_create_domain_token_scoped_with_domain_id_and_user_id(self):
        path = '/domains/%s/users/%s/roles/%s' % (self.domain['id'], self.user['id'], self.role['id'])
        self.put(path=path)
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_id=self.domain['id'])
        r = self.v3_create_token(auth_data)
        self.assertValidDomainScopedTokenResponse(r)

    def test_create_domain_token_scoped_with_domain_id_and_username(self):
        path = '/domains/%s/users/%s/roles/%s' % (self.domain['id'], self.user['id'], self.role['id'])
        self.put(path=path)
        auth_data = self.build_authentication_request(username=self.user['name'], user_domain_id=self.domain['id'], password=self.user['password'], domain_id=self.domain['id'])
        r = self.v3_create_token(auth_data)
        self.assertValidDomainScopedTokenResponse(r)

    def test_create_domain_token_scoped_with_domain_id(self):
        path = '/domains/%s/users/%s/roles/%s' % (self.domain['id'], self.user['id'], self.role['id'])
        self.put(path=path)
        auth_data = self.build_authentication_request(username=self.user['name'], user_domain_name=self.domain['name'], password=self.user['password'], domain_id=self.domain['id'])
        r = self.v3_create_token(auth_data)
        self.assertValidDomainScopedTokenResponse(r)

    def test_create_domain_token_scoped_with_domain_name(self):
        path = '/domains/%s/users/%s/roles/%s' % (self.domain['id'], self.user['id'], self.role['id'])
        self.put(path=path)
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_name=self.domain['name'])
        r = self.v3_create_token(auth_data)
        self.assertValidDomainScopedTokenResponse(r)

    def test_create_domain_token_scoped_with_domain_name_and_username(self):
        path = '/domains/%s/users/%s/roles/%s' % (self.domain['id'], self.user['id'], self.role['id'])
        self.put(path=path)
        auth_data = self.build_authentication_request(username=self.user['name'], user_domain_id=self.domain['id'], password=self.user['password'], domain_name=self.domain['name'])
        r = self.v3_create_token(auth_data)
        self.assertValidDomainScopedTokenResponse(r)

    def test_create_domain_token_with_only_domain_name_and_username(self):
        path = '/domains/%s/users/%s/roles/%s' % (self.domain['id'], self.user['id'], self.role['id'])
        self.put(path=path)
        auth_data = self.build_authentication_request(username=self.user['name'], user_domain_name=self.domain['name'], password=self.user['password'], domain_name=self.domain['name'])
        r = self.v3_create_token(auth_data)
        self.assertValidDomainScopedTokenResponse(r)

    def test_create_domain_token_with_group_role(self):
        group = unit.new_group_ref(domain_id=self.domain_id)
        group = PROVIDERS.identity_api.create_group(group)
        PROVIDERS.identity_api.add_user_to_group(self.user['id'], group['id'])
        path = '/domains/%s/groups/%s/roles/%s' % (self.domain['id'], group['id'], self.role['id'])
        self.put(path=path)
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_id=self.domain['id'])
        r = self.v3_create_token(auth_data)
        self.assertValidDomainScopedTokenResponse(r)

    def test_create_domain_token_fails_if_domain_name_unsafe(self):
        """Verify authenticate to a domain with unsafe name fails."""
        self.config_fixture.config(group='resource', domain_name_url_safe='off')
        unsafe_name = 'i am not / safe'
        domain = unit.new_domain_ref(name=unsafe_name)
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        role_member = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_member['id'], role_member)
        PROVIDERS.assignment_api.create_grant(role_member['id'], user_id=self.user['id'], domain_id=domain['id'])
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_name=domain['name'])
        self.v3_create_token(auth_data)
        self.config_fixture.config(group='resource', project_name_url_safe='new')
        self.v3_create_token(auth_data)
        self.config_fixture.config(group='resource', domain_name_url_safe='strict')
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_create_domain_token_without_grant_returns_unauthorized(self):
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_id=self.domain['id'])
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_validate_domain_scoped_token(self):
        PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=self.user['id'], domain_id=self.domain['id'])
        domain_scoped_token = self._get_domain_scoped_token()
        r = self._validate_token(domain_scoped_token)
        self.assertValidDomainScopedTokenResponse(r)
        resp_json = json.loads(r.body)
        self.assertIsNotNone(resp_json['token']['catalog'])
        self.assertIsNotNone(resp_json['token']['roles'])
        self.assertIsNotNone(resp_json['token']['domain'])

    def test_validate_expired_domain_scoped_token_returns_not_found(self):
        PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=self.user['id'], domain_id=self.domain['id'])
        self.config_fixture.config(group='token', expiration=10)
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            domain_scoped_token = self._get_domain_scoped_token()
            frozen_datetime.tick(delta=datetime.timedelta(seconds=15))
            self._validate_token(domain_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_domain_scoped_token_is_invalid_after_disabling_user(self):
        PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=self.user['id'], domain_id=self.domain['id'])
        domain_scoped_token = self._get_domain_scoped_token()
        r = self._validate_token(domain_scoped_token)
        self.assertValidDomainScopedTokenResponse(r)
        self._set_user_enabled(self.user, enabled=False)
        self._validate_token(domain_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_domain_scoped_token_is_invalid_after_deleting_grant(self):
        PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=self.user['id'], domain_id=self.domain['id'])
        domain_scoped_token = self._get_domain_scoped_token()
        r = self._validate_token(domain_scoped_token)
        self.assertValidDomainScopedTokenResponse(r)
        PROVIDERS.assignment_api.delete_grant(self.role['id'], user_id=self.user['id'], domain_id=self.domain['id'])
        self._validate_token(domain_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_domain_scoped_token_invalid_after_disabling_domain(self):
        PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=self.user['id'], domain_id=self.domain['id'])
        domain_scoped_token = self._get_domain_scoped_token()
        r = self._validate_token(domain_scoped_token)
        self.assertValidDomainScopedTokenResponse(r)
        self.domain['enabled'] = False
        PROVIDERS.resource_api.update_domain(self.domain['id'], self.domain)
        self._validate_token(domain_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_create_project_scoped_token_with_project_id_and_user_id(self):
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        r = self.v3_create_token(auth_data)
        self.assertValidProjectScopedTokenResponse(r)

    def test_validate_project_scoped_token(self):
        project_scoped_token = self._get_project_scoped_token()
        r = self._validate_token(project_scoped_token)
        self.assertValidProjectScopedTokenResponse(r)

    def test_validate_expired_project_scoped_token_returns_not_found(self):
        self.config_fixture.config(group='token', expiration=10)
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            project_scoped_token = self._get_project_scoped_token()
            frozen_datetime.tick(delta=datetime.timedelta(seconds=15))
            self._validate_token(project_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_revoke_project_scoped_token(self):
        project_scoped_token = self._get_project_scoped_token()
        r = self._validate_token(project_scoped_token)
        self.assertValidProjectScopedTokenResponse(r)
        self._revoke_token(project_scoped_token)
        self._validate_token(project_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_project_scoped_token_is_scoped_to_default_project(self):
        project = self._create_project_and_set_as_default_project()
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'])
        r = self.v3_create_token(auth_data)
        self.assertValidProjectScopedTokenResponse(r)
        self.assertEqual(project['id'], r.result['token']['project']['id'])

    def test_project_scoped_token_no_catalog_is_scoped_to_default_project(self):
        project = self._create_project_and_set_as_default_project()
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'])
        r = self.post('/auth/tokens?nocatalog', body=auth_data, noauth=True)
        self.assertValidProjectScopedTokenResponse(r, require_catalog=False)
        self.assertEqual(project['id'], r.result['token']['project']['id'])

    def test_implicit_project_id_scoped_token_with_user_id_no_catalog(self):
        self._create_project_and_set_as_default_project()
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        r = self.post('/auth/tokens?nocatalog', body=auth_data, noauth=True)
        self.assertValidProjectScopedTokenResponse(r, require_catalog=False)
        self.assertEqual(self.project['id'], r.result['token']['project']['id'])

    def test_project_scoped_token_catalog_attributes(self):
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        r = self.v3_create_token(auth_data)
        catalog = r.result['token']['catalog']
        self.assertEqual(1, len(catalog))
        catalog = catalog[0]
        self.assertEqual(self.service['id'], catalog['id'])
        self.assertEqual(self.service['name'], catalog['name'])
        self.assertEqual(self.service['type'], catalog['type'])
        endpoint = catalog['endpoints']
        self.assertEqual(1, len(endpoint))
        endpoint = endpoint[0]
        self.assertEqual(self.endpoint['id'], endpoint['id'])
        self.assertEqual(self.endpoint['interface'], endpoint['interface'])
        self.assertEqual(self.endpoint['region_id'], endpoint['region_id'])
        self.assertEqual(self.endpoint['url'], endpoint['url'])

    def test_project_scoped_token_catalog_excludes_disabled_endpoint(self):
        disabled_endpoint_ref = copy.copy(self.endpoint)
        disabled_endpoint_id = uuid.uuid4().hex
        disabled_endpoint_ref.update({'id': disabled_endpoint_id, 'enabled': False, 'interface': 'internal'})
        PROVIDERS.catalog_api.create_endpoint(disabled_endpoint_id, disabled_endpoint_ref)
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        resp = self.v3_create_token(auth_data)
        endpoints = resp.result['token']['catalog'][0]['endpoints']
        endpoint_ids = [endpoint['id'] for endpoint in endpoints]
        self.assertNotIn(disabled_endpoint_id, endpoint_ids)

    def test_project_scoped_token_catalog_excludes_disabled_service(self):
        """On authenticate, get a catalog that excludes disabled services."""
        self.assertTrue(self.endpoint['enabled'])
        PROVIDERS.catalog_api.update_service(self.endpoint['service_id'], {'enabled': False})
        service = PROVIDERS.catalog_api.get_service(self.endpoint['service_id'])
        self.assertFalse(service['enabled'])
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        r = self.v3_create_token(auth_data)
        self.assertEqual([], r.result['token']['catalog'])

    def test_scope_to_project_without_grant_returns_unauthorized(self):
        project = unit.new_project_ref(domain_id=self.domain_id)
        PROVIDERS.resource_api.create_project(project['id'], project)
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=project['id'])
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_create_project_scoped_token_with_username_and_domain_id(self):
        auth_data = self.build_authentication_request(username=self.user['name'], user_domain_id=self.domain['id'], password=self.user['password'], project_id=self.project['id'])
        r = self.v3_create_token(auth_data)
        self.assertValidProjectScopedTokenResponse(r)

    def test_create_project_scoped_token_with_username_and_domain_name(self):
        auth_data = self.build_authentication_request(username=self.user['name'], user_domain_name=self.domain['name'], password=self.user['password'], project_id=self.project['id'])
        r = self.v3_create_token(auth_data)
        self.assertValidProjectScopedTokenResponse(r)

    def test_create_project_scoped_token_fails_if_project_name_unsafe(self):
        """Verify authenticate to a project with unsafe name fails."""
        self.config_fixture.config(group='resource', project_name_url_safe='off')
        unsafe_name = 'i am not / safe'
        project = unit.new_project_ref(domain_id=test_v3.DEFAULT_DOMAIN_ID, name=unsafe_name)
        PROVIDERS.resource_api.create_project(project['id'], project)
        role_member = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_member['id'], role_member)
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.user['id'], project['id'], role_member['id'])
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_name=project['name'], project_domain_id=test_v3.DEFAULT_DOMAIN_ID)
        self.v3_create_token(auth_data)
        self.config_fixture.config(group='resource', project_name_url_safe='new')
        self.v3_create_token(auth_data)
        self.config_fixture.config(group='resource', project_name_url_safe='strict')
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_create_project_scoped_token_fails_if_domain_name_unsafe(self):
        """Verify authenticate to a project using unsafe domain name fails."""
        self.config_fixture.config(group='resource', domain_name_url_safe='off')
        unsafe_name = 'i am not / safe'
        domain = unit.new_domain_ref(name=unsafe_name)
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        project = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project['id'], project)
        role_member = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_member['id'], role_member)
        PROVIDERS.assignment_api.create_grant(role_member['id'], user_id=self.user['id'], project_id=project['id'])
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_name=project['name'], project_domain_name=domain['name'])
        self.v3_create_token(auth_data)
        self.config_fixture.config(group='resource', project_name_url_safe='new')
        self.v3_create_token(auth_data)
        self.config_fixture.config(group='resource', domain_name_url_safe='strict')
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_create_project_token_with_same_domain_and_project_name(self):
        """Authenticate to a project with the same name as its domain."""
        domain = unit.new_project_ref(is_domain=True)
        domain = PROVIDERS.resource_api.create_project(domain['id'], domain)
        project = unit.new_project_ref(domain_id=domain['id'], name=domain['name'])
        PROVIDERS.resource_api.create_project(project['id'], project)
        role_member = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_member['id'], role_member)
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.user['id'], project['id'], role_member['id'])
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_name=project['name'], project_domain_name=domain['name'])
        r = self.v3_create_token(auth_data)
        self.assertEqual(project['id'], r.result['token']['project']['id'])

    def test_create_project_token_fails_with_project_acting_as_domain(self):
        domain = unit.new_project_ref(is_domain=True)
        domain = PROVIDERS.resource_api.create_project(domain['id'], domain)
        role_member = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_member['id'], role_member)
        PROVIDERS.assignment_api.create_grant(role_member['id'], user_id=self.user['id'], domain_id=domain['id'])
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_name=domain['name'], project_domain_name=domain['name'])
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_create_project_token_with_disabled_project_domain_fails(self):
        domain = unit.new_domain_ref()
        domain = PROVIDERS.resource_api.create_domain(domain['id'], domain)
        project = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project['id'], project)
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.user['id'], project['id'], self.role_id)
        domain['enabled'] = False
        PROVIDERS.resource_api.update_domain(domain['id'], domain)
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=project['id'])
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_name=project['name'], project_domain_id=domain['id'])
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_create_project_token_with_default_domain_as_project(self):
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=test_v3.DEFAULT_DOMAIN_ID)
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_project_scoped_token_is_invalid_after_disabling_user(self):
        project_scoped_token = self._get_project_scoped_token()
        r = self._validate_token(project_scoped_token)
        self.assertValidProjectScopedTokenResponse(r)
        self._set_user_enabled(self.user, enabled=False)
        self._validate_token(project_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_project_scoped_token_invalid_after_changing_user_password(self):
        project_scoped_token = self._get_project_scoped_token()
        r = self._validate_token(project_scoped_token)
        self.assertValidProjectScopedTokenResponse(r)
        self.user['password'] = 'Password1'
        PROVIDERS.identity_api.update_user(self.user['id'], self.user)
        self._validate_token(project_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_project_scoped_token_invalid_after_disabling_project(self):
        project_scoped_token = self._get_project_scoped_token()
        r = self._validate_token(project_scoped_token)
        self.assertValidProjectScopedTokenResponse(r)
        self.project['enabled'] = False
        PROVIDERS.resource_api.update_project(self.project['id'], self.project)
        self._validate_token(project_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_project_scoped_token_is_invalid_after_deleting_grant(self):
        self.config_fixture.config(group='cache', enabled=False)
        PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=self.user['id'], project_id=self.project['id'])
        project_scoped_token = self._get_project_scoped_token()
        r = self._validate_token(project_scoped_token)
        self.assertValidProjectScopedTokenResponse(r)
        PROVIDERS.assignment_api.delete_grant(self.role['id'], user_id=self.user['id'], project_id=self.project['id'])
        self._validate_token(project_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_no_access_to_default_project_result_in_unscoped_token(self):
        self.create_new_default_project_for_user(self.user['id'], self.domain_id)
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'])
        r = self.v3_create_token(auth_data)
        self.assertValidUnscopedTokenResponse(r)

    def test_rescope_unscoped_token_with_trust(self):
        trustee_user, trust = self._create_trust()
        self._get_trust_scoped_token(trustee_user, trust)

    def test_validate_a_trust_scoped_token(self):
        trustee_user, trust = self._create_trust()
        trust_scoped_token = self._get_trust_scoped_token(trustee_user, trust)
        r = self._validate_token(trust_scoped_token)
        self.assertValidProjectScopedTokenResponse(r)

    def test_validate_expired_trust_scoped_token_returns_not_found(self):
        self.config_fixture.config(group='token', expiration=10)
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            trustee_user, trust = self._create_trust()
            trust_scoped_token = self._get_trust_scoped_token(trustee_user, trust)
            frozen_datetime.tick(delta=datetime.timedelta(seconds=15))
            self._validate_token(trust_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_validate_a_trust_scoped_token_impersonated(self):
        trustee_user, trust = self._create_trust(impersonation=True)
        trust_scoped_token = self._get_trust_scoped_token(trustee_user, trust)
        r = self._validate_token(trust_scoped_token)
        self.assertValidProjectScopedTokenResponse(r)

    def test_revoke_trust_scoped_token(self):
        trustee_user, trust = self._create_trust()
        trust_scoped_token = self._get_trust_scoped_token(trustee_user, trust)
        r = self._validate_token(trust_scoped_token)
        self.assertValidProjectScopedTokenResponse(r)
        self._revoke_token(trust_scoped_token)
        self._validate_token(trust_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_trust_scoped_token_is_invalid_after_disabling_trustee(self):
        trustee_user, trust = self._create_trust()
        trust_scoped_token = self._get_trust_scoped_token(trustee_user, trust)
        r = self._validate_token(trust_scoped_token)
        self.assertValidProjectScopedTokenResponse(r)
        trustee_update_ref = dict(enabled=False)
        PROVIDERS.identity_api.update_user(trustee_user['id'], trustee_update_ref)
        self._validate_token(trust_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_trust_token_is_invalid_when_trustee_domain_disabled(self):
        new_domain_ref = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(new_domain_ref['id'], new_domain_ref)
        trustee_ref = unit.create_user(PROVIDERS.identity_api, domain_id=new_domain_ref['id'])
        new_project_ref = unit.new_project_ref(domain_id=self.domain_id)
        PROVIDERS.resource_api.create_project(new_project_ref['id'], new_project_ref)
        PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=self.user_id, project_id=new_project_ref['id'])
        trust_ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=trustee_ref['id'], expires=dict(minutes=1), project_id=new_project_ref['id'], impersonation=True, role_ids=[self.role['id']])
        resp = self.post('/OS-TRUST/trusts', body={'trust': trust_ref})
        self.assertValidTrustResponse(resp, trust_ref)
        trust_id = resp.json_body['trust']['id']
        trust_auth_data = self.build_authentication_request(user_id=trustee_ref['id'], password=trustee_ref['password'], trust_id=trust_id)
        trust_scoped_token = self.get_requested_token(trust_auth_data)
        self._validate_token(trust_scoped_token)
        disable_body = {'domain': {'enabled': False}}
        self.patch('/domains/%(domain_id)s' % {'domain_id': new_domain_ref['id']}, body=disable_body)
        self._validate_token(trust_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_trust_scoped_token_invalid_after_changing_trustee_password(self):
        trustee_user, trust = self._create_trust()
        trust_scoped_token = self._get_trust_scoped_token(trustee_user, trust)
        r = self._validate_token(trust_scoped_token)
        self.assertValidProjectScopedTokenResponse(r)
        trustee_update_ref = dict(password='Password1')
        PROVIDERS.identity_api.update_user(trustee_user['id'], trustee_update_ref)
        self._validate_token(trust_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_trust_scoped_token_is_invalid_after_disabling_trustor(self):
        trustee_user, trust = self._create_trust()
        trust_scoped_token = self._get_trust_scoped_token(trustee_user, trust)
        r = self._validate_token(trust_scoped_token)
        self.assertValidProjectScopedTokenResponse(r)
        trustor_update_ref = dict(enabled=False)
        PROVIDERS.identity_api.update_user(self.user['id'], trustor_update_ref)
        self._validate_token(trust_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_trust_scoped_token_invalid_after_changing_trustor_password(self):
        trustee_user, trust = self._create_trust()
        trust_scoped_token = self._get_trust_scoped_token(trustee_user, trust)
        r = self._validate_token(trust_scoped_token)
        self.assertValidProjectScopedTokenResponse(r)
        trustor_update_ref = dict(password='Password1')
        PROVIDERS.identity_api.update_user(self.user['id'], trustor_update_ref)
        self._validate_token(trust_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_trust_scoped_token_invalid_after_disabled_trustor_domain(self):
        trustee_user, trust = self._create_trust()
        trust_scoped_token = self._get_trust_scoped_token(trustee_user, trust)
        r = self._validate_token(trust_scoped_token)
        self.assertValidProjectScopedTokenResponse(r)
        self.domain['enabled'] = False
        PROVIDERS.resource_api.update_domain(self.domain['id'], self.domain)
        trustor_update_ref = dict(password='Password1')
        PROVIDERS.identity_api.update_user(self.user['id'], trustor_update_ref)
        self._validate_token(trust_scoped_token, expected_status=http.client.NOT_FOUND)

    def test_default_fixture_scope_token(self):
        self.assertIsNotNone(self.get_scoped_token())

    def test_rescoping_token(self):
        expires = self.v3_token_data['token']['expires_at']
        r = self.v3_create_token(self.build_authentication_request(token=self.v3_token, project_id=self.project_id))
        self.assertValidProjectScopedTokenResponse(r)
        self.assertTimestampEqual(expires, r.result['token']['expires_at'])

    def test_check_token(self):
        self.head('/auth/tokens', headers=self.headers, expected_status=http.client.OK)

    def test_validate_token(self):
        r = self.get('/auth/tokens', headers=self.headers)
        self.assertValidUnscopedTokenResponse(r)

    def test_validate_missing_subject_token(self):
        self.get('/auth/tokens', expected_status=http.client.NOT_FOUND)

    def test_validate_missing_auth_token(self):
        self.admin_request(method='GET', path='/v3/projects', token=None, expected_status=http.client.UNAUTHORIZED)

    def test_validate_token_nocatalog(self):
        v3_token = self.get_requested_token(self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id']))
        r = self.get('/auth/tokens?nocatalog', headers={'X-Subject-Token': v3_token})
        self.assertValidProjectScopedTokenResponse(r, require_catalog=False)

    def test_is_admin_token_by_ids(self):
        self.config_fixture.config(group='resource', admin_project_domain_name=self.domain['name'], admin_project_name=self.project['name'])
        r = self.v3_create_token(self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id']))
        self.assertValidProjectScopedTokenResponse(r, is_admin_project=True)
        v3_token = r.headers.get('X-Subject-Token')
        r = self.get('/auth/tokens', headers={'X-Subject-Token': v3_token})
        self.assertValidProjectScopedTokenResponse(r, is_admin_project=True)

    def test_is_admin_token_by_names(self):
        self.config_fixture.config(group='resource', admin_project_domain_name=self.domain['name'], admin_project_name=self.project['name'])
        r = self.v3_create_token(self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_domain_name=self.domain['name'], project_name=self.project['name']))
        self.assertValidProjectScopedTokenResponse(r, is_admin_project=True)
        v3_token = r.headers.get('X-Subject-Token')
        r = self.get('/auth/tokens', headers={'X-Subject-Token': v3_token})
        self.assertValidProjectScopedTokenResponse(r, is_admin_project=True)

    def test_token_for_non_admin_project_is_not_admin(self):
        self.config_fixture.config(group='resource', admin_project_domain_name=self.domain['name'], admin_project_name=uuid.uuid4().hex)
        r = self.v3_create_token(self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id']))
        self.assertValidProjectScopedTokenResponse(r, is_admin_project=False)
        v3_token = r.headers.get('X-Subject-Token')
        r = self.get('/auth/tokens', headers={'X-Subject-Token': v3_token})
        self.assertValidProjectScopedTokenResponse(r, is_admin_project=False)

    def test_token_for_non_admin_domain_same_project_name_is_not_admin(self):
        self.config_fixture.config(group='resource', admin_project_domain_name=uuid.uuid4().hex, admin_project_name=self.project['name'])
        r = self.v3_create_token(self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id']))
        self.assertValidProjectScopedTokenResponse(r, is_admin_project=False)
        v3_token = r.headers.get('X-Subject-Token')
        r = self.get('/auth/tokens', headers={'X-Subject-Token': v3_token})
        self.assertValidProjectScopedTokenResponse(r, is_admin_project=False)

    def test_only_admin_project_set_acts_as_non_admin(self):
        self.config_fixture.config(group='resource', admin_project_name=self.project['name'])
        r = self.v3_create_token(self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id']))
        self.assertValidProjectScopedTokenResponse(r, is_admin_project=None)
        v3_token = r.headers.get('X-Subject-Token')
        r = self.get('/auth/tokens', headers={'X-Subject-Token': v3_token})
        self.assertValidProjectScopedTokenResponse(r, is_admin_project=None)

    def _create_role(self, domain_id=None):
        """Call ``POST /roles``."""
        ref = unit.new_role_ref(domain_id=domain_id)
        r = self.post('/roles', body={'role': ref})
        return self.assertValidRoleResponse(r, ref)

    def _create_implied_role(self, prior_id):
        implied = self._create_role()
        url = '/roles/%s/implies/%s' % (prior_id, implied['id'])
        self.put(url, expected_status=http.client.CREATED)
        return implied

    def _delete_implied_role(self, prior_role_id, implied_role_id):
        url = '/roles/%s/implies/%s' % (prior_role_id, implied_role_id)
        self.delete(url)

    def _get_scoped_token_roles(self, is_domain=False):
        if is_domain:
            v3_token = self.get_domain_scoped_token()
        else:
            v3_token = self.get_scoped_token()
        r = self.get('/auth/tokens', headers={'X-Subject-Token': v3_token})
        v3_token_data = r.result
        token_roles = v3_token_data['token']['roles']
        return token_roles

    def _create_implied_role_shows_in_v3_token(self, is_domain):
        token_roles = self._get_scoped_token_roles(is_domain)
        self.assertEqual(1, len(token_roles))
        prior = token_roles[0]['id']
        implied1 = self._create_implied_role(prior)
        token_roles = self._get_scoped_token_roles(is_domain)
        self.assertEqual(2, len(token_roles))
        implied2 = self._create_implied_role(prior)
        token_roles = self._get_scoped_token_roles(is_domain)
        self.assertEqual(3, len(token_roles))
        token_role_ids = [role['id'] for role in token_roles]
        self.assertIn(prior, token_role_ids)
        self.assertIn(implied1['id'], token_role_ids)
        self.assertIn(implied2['id'], token_role_ids)

    def test_create_implied_role_shows_in_v3_project_token(self):
        self.config_fixture.config(group='token')
        self._create_implied_role_shows_in_v3_token(False)

    def test_create_implied_role_shows_in_v3_domain_token(self):
        self.config_fixture.config(group='token')
        PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=self.user['id'], domain_id=self.domain['id'])
        self._create_implied_role_shows_in_v3_token(True)

    def test_create_implied_role_shows_in_v3_system_token(self):
        self.config_fixture.config(group='token')
        PROVIDERS.assignment_api.create_system_grant_for_user(self.user['id'], self.role['id'])
        token_id = self.get_system_scoped_token()
        r = self.get('/auth/tokens', headers={'X-Subject-Token': token_id})
        token_roles = r.result['token']['roles']
        prior = token_roles[0]['id']
        self._create_implied_role(prior)
        r = self.get('/auth/tokens', headers={'X-Subject-Token': token_id})
        token_roles = r.result['token']['roles']
        self.assertEqual(2, len(token_roles))

    def test_group_assigned_implied_role_shows_in_v3_token(self):
        self.config_fixture.config(group='token')
        is_domain = False
        token_roles = self._get_scoped_token_roles(is_domain)
        self.assertEqual(1, len(token_roles))
        new_role = self._create_role()
        prior = new_role['id']
        new_group_ref = unit.new_group_ref(domain_id=self.domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group_ref)
        PROVIDERS.assignment_api.create_grant(prior, group_id=new_group['id'], project_id=self.project['id'])
        token_roles = self._get_scoped_token_roles(is_domain)
        self.assertEqual(1, len(token_roles))
        PROVIDERS.identity_api.add_user_to_group(self.user['id'], new_group['id'])
        token_roles = self._get_scoped_token_roles(is_domain)
        self.assertEqual(2, len(token_roles))
        implied1 = self._create_implied_role(prior)
        token_roles = self._get_scoped_token_roles(is_domain)
        self.assertEqual(3, len(token_roles))
        implied2 = self._create_implied_role(prior)
        token_roles = self._get_scoped_token_roles(is_domain)
        self.assertEqual(4, len(token_roles))
        token_role_ids = [role['id'] for role in token_roles]
        self.assertIn(prior, token_role_ids)
        self.assertIn(implied1['id'], token_role_ids)
        self.assertIn(implied2['id'], token_role_ids)

    def test_multiple_implied_roles_show_in_v3_token(self):
        self.config_fixture.config(group='token')
        token_roles = self._get_scoped_token_roles()
        self.assertEqual(1, len(token_roles))
        prior = token_roles[0]['id']
        implied1 = self._create_implied_role(prior)
        implied2 = self._create_implied_role(prior)
        implied3 = self._create_implied_role(prior)
        token_roles = self._get_scoped_token_roles()
        self.assertEqual(4, len(token_roles))
        token_role_ids = [role['id'] for role in token_roles]
        self.assertIn(prior, token_role_ids)
        self.assertIn(implied1['id'], token_role_ids)
        self.assertIn(implied2['id'], token_role_ids)
        self.assertIn(implied3['id'], token_role_ids)

    def test_chained_implied_role_shows_in_v3_token(self):
        self.config_fixture.config(group='token')
        token_roles = self._get_scoped_token_roles()
        self.assertEqual(1, len(token_roles))
        prior = token_roles[0]['id']
        implied1 = self._create_implied_role(prior)
        implied2 = self._create_implied_role(implied1['id'])
        implied3 = self._create_implied_role(implied2['id'])
        token_roles = self._get_scoped_token_roles()
        self.assertEqual(4, len(token_roles))
        token_role_ids = [role['id'] for role in token_roles]
        self.assertIn(prior, token_role_ids)
        self.assertIn(implied1['id'], token_role_ids)
        self.assertIn(implied2['id'], token_role_ids)
        self.assertIn(implied3['id'], token_role_ids)

    def test_implied_role_disabled_by_config(self):
        self.config_fixture.config(group='token')
        token_roles = self._get_scoped_token_roles()
        self.assertEqual(1, len(token_roles))
        prior = token_roles[0]['id']
        implied1 = self._create_implied_role(prior)
        implied2 = self._create_implied_role(implied1['id'])
        self._create_implied_role(implied2['id'])
        token_roles = self._get_scoped_token_roles()
        self.assertEqual(4, len(token_roles))
        token_role_ids = [role['id'] for role in token_roles]
        self.assertIn(prior, token_role_ids)

    def test_delete_implied_role_do_not_show_in_v3_token(self):
        self.config_fixture.config(group='token')
        token_roles = self._get_scoped_token_roles()
        prior = token_roles[0]['id']
        implied = self._create_implied_role(prior)
        token_roles = self._get_scoped_token_roles()
        self.assertEqual(2, len(token_roles))
        self._delete_implied_role(prior, implied['id'])
        token_roles = self._get_scoped_token_roles()
        self.assertEqual(1, len(token_roles))

    def test_unrelated_implied_roles_do_not_change_v3_token(self):
        self.config_fixture.config(group='token')
        token_roles = self._get_scoped_token_roles()
        prior = token_roles[0]['id']
        implied = self._create_implied_role(prior)
        token_roles = self._get_scoped_token_roles()
        self.assertEqual(2, len(token_roles))
        unrelated = self._create_role()
        url = '/roles/%s/implies/%s' % (unrelated['id'], implied['id'])
        self.put(url, expected_status=http.client.CREATED)
        token_roles = self._get_scoped_token_roles()
        self.assertEqual(2, len(token_roles))
        self._delete_implied_role(unrelated['id'], implied['id'])
        token_roles = self._get_scoped_token_roles()
        self.assertEqual(2, len(token_roles))

    def test_domain_specific_roles_do_not_show_v3_token(self):
        self.config_fixture.config(group='token')
        initial_token_roles = self._get_scoped_token_roles()
        new_role = self._create_role(domain_id=self.domain_id)
        PROVIDERS.assignment_api.create_grant(new_role['id'], user_id=self.user['id'], project_id=self.project['id'])
        implied = self._create_implied_role(new_role['id'])
        token_roles = self._get_scoped_token_roles()
        self.assertEqual(len(initial_token_roles) + 1, len(token_roles))
        token_role_ids = [role['id'] for role in token_roles]
        self.assertIn(implied['id'], token_role_ids)
        self.assertNotIn(new_role['id'], token_role_ids)

    def test_remove_all_roles_from_scope_result_in_404(self):
        new_user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        path = '/projects/%s/users/%s/roles/%s' % (self.project['id'], new_user['id'], self.role['id'])
        self.put(path=path)
        auth_data = self.build_authentication_request(user_id=new_user['id'], password=new_user['password'], project_id=self.project['id'])
        subject_token_id = self.v3_create_token(auth_data).headers.get('X-Subject-Token')
        headers = {'X-Subject-Token': subject_token_id}
        r = self.get('/auth/tokens', headers=headers)
        self.assertValidProjectScopedTokenResponse(r)
        path = '/projects/%s/users/%s/roles/%s' % (self.project['id'], new_user['id'], self.role['id'])
        self.delete(path=path)
        self.get('/auth/tokens', headers=headers, expected_status=http.client.NOT_FOUND)

    def test_create_token_with_nonexistant_user_id_fails(self):
        auth_data = self.build_authentication_request(user_id=uuid.uuid4().hex, password=self.user['password'])
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_create_token_with_nonexistant_username_fails(self):
        auth_data = self.build_authentication_request(username=uuid.uuid4().hex, user_domain_id=self.domain['id'], password=self.user['password'])
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_create_token_with_nonexistant_domain_id_fails(self):
        auth_data = self.build_authentication_request(username=self.user['name'], user_domain_id=uuid.uuid4().hex, password=self.user['password'])
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_create_token_with_nonexistant_domain_name_fails(self):
        auth_data = self.build_authentication_request(username=self.user['name'], user_domain_name=uuid.uuid4().hex, password=self.user['password'])
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_create_token_with_wrong_password_fails(self):
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=uuid.uuid4().hex)
        self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_user_and_group_roles_scoped_token(self):
        """Test correct roles are returned in scoped token.

        Test Plan:

        - Create a domain, with 1 project, 2 users (user1 and user2)
          and 2 groups (group1 and group2)
        - Make user1 a member of group1, user2 a member of group2
        - Create 8 roles, assigning them to each of the 8 combinations
          of users/groups on domain/project
        - Get a project scoped token for user1, checking that the right
          two roles are returned (one directly assigned, one by virtue
          of group membership)
        - Repeat this for a domain scoped token
        - Make user1 also a member of group2
        - Get another scoped token making sure the additional role
          shows up
        - User2 is just here as a spoiler, to make sure we don't get
          any roles uniquely assigned to it returned in any of our
          tokens

        """
        domainA = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domainA['id'], domainA)
        projectA = unit.new_project_ref(domain_id=domainA['id'])
        PROVIDERS.resource_api.create_project(projectA['id'], projectA)
        user1 = unit.create_user(PROVIDERS.identity_api, domain_id=domainA['id'])
        user2 = unit.create_user(PROVIDERS.identity_api, domain_id=domainA['id'])
        group1 = unit.new_group_ref(domain_id=domainA['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        group2 = unit.new_group_ref(domain_id=domainA['id'])
        group2 = PROVIDERS.identity_api.create_group(group2)
        PROVIDERS.identity_api.add_user_to_group(user1['id'], group1['id'])
        PROVIDERS.identity_api.add_user_to_group(user2['id'], group2['id'])
        role_list = []
        for _ in range(8):
            role = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role['id'], role)
            role_list.append(role)
        PROVIDERS.assignment_api.create_grant(role_list[0]['id'], user_id=user1['id'], domain_id=domainA['id'])
        PROVIDERS.assignment_api.create_grant(role_list[1]['id'], user_id=user1['id'], project_id=projectA['id'])
        PROVIDERS.assignment_api.create_grant(role_list[2]['id'], user_id=user2['id'], domain_id=domainA['id'])
        PROVIDERS.assignment_api.create_grant(role_list[3]['id'], user_id=user2['id'], project_id=projectA['id'])
        PROVIDERS.assignment_api.create_grant(role_list[4]['id'], group_id=group1['id'], domain_id=domainA['id'])
        PROVIDERS.assignment_api.create_grant(role_list[5]['id'], group_id=group1['id'], project_id=projectA['id'])
        PROVIDERS.assignment_api.create_grant(role_list[6]['id'], group_id=group2['id'], domain_id=domainA['id'])
        PROVIDERS.assignment_api.create_grant(role_list[7]['id'], group_id=group2['id'], project_id=projectA['id'])
        auth_data = self.build_authentication_request(user_id=user1['id'], password=user1['password'], project_id=projectA['id'])
        r = self.v3_create_token(auth_data)
        token = self.assertValidScopedTokenResponse(r)
        roles_ids = []
        for ref in token['roles']:
            roles_ids.append(ref['id'])
        self.assertEqual(2, len(token['roles']))
        self.assertIn(role_list[1]['id'], roles_ids)
        self.assertIn(role_list[5]['id'], roles_ids)
        auth_data = self.build_authentication_request(user_id=user1['id'], password=user1['password'], domain_id=domainA['id'])
        r = self.v3_create_token(auth_data)
        token = self.assertValidScopedTokenResponse(r)
        roles_ids = []
        for ref in token['roles']:
            roles_ids.append(ref['id'])
        self.assertEqual(2, len(token['roles']))
        self.assertIn(role_list[0]['id'], roles_ids)
        self.assertIn(role_list[4]['id'], roles_ids)
        PROVIDERS.identity_api.add_user_to_group(user1['id'], group2['id'])
        auth_data = self.build_authentication_request(user_id=user1['id'], password=user1['password'], project_id=projectA['id'])
        r = self.v3_create_token(auth_data)
        token = self.assertValidScopedTokenResponse(r)
        roles_ids = []
        for ref in token['roles']:
            roles_ids.append(ref['id'])
        self.assertEqual(3, len(token['roles']))
        self.assertIn(role_list[1]['id'], roles_ids)
        self.assertIn(role_list[5]['id'], roles_ids)
        self.assertIn(role_list[7]['id'], roles_ids)

    def test_auth_token_cross_domain_group_and_project(self):
        """Verify getting a token in cross domain group/project roles."""
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        project1 = unit.new_project_ref(domain_id=domain1['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        user_foo = unit.create_user(PROVIDERS.identity_api, domain_id=test_v3.DEFAULT_DOMAIN_ID)
        role_member = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_member['id'], role_member)
        role_admin = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_admin['id'], role_admin)
        role_foo_domain1 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_foo_domain1['id'], role_foo_domain1)
        role_group_domain1 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_group_domain1['id'], role_group_domain1)
        new_group = unit.new_group_ref(domain_id=domain1['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        PROVIDERS.identity_api.add_user_to_group(user_foo['id'], new_group['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user_foo['id'], project_id=project1['id'], role_id=role_member['id'])
        PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], project_id=project1['id'], role_id=role_admin['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user_foo['id'], domain_id=domain1['id'], role_id=role_foo_domain1['id'])
        PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], domain_id=domain1['id'], role_id=role_group_domain1['id'])
        auth_data = self.build_authentication_request(username=user_foo['name'], user_domain_id=test_v3.DEFAULT_DOMAIN_ID, password=user_foo['password'], project_name=project1['name'], project_domain_id=domain1['id'])
        r = self.v3_create_token(auth_data)
        scoped_token = self.assertValidScopedTokenResponse(r)
        project = scoped_token['project']
        roles_ids = []
        for ref in scoped_token['roles']:
            roles_ids.append(ref['id'])
        self.assertEqual(project1['id'], project['id'])
        self.assertIn(role_member['id'], roles_ids)
        self.assertIn(role_admin['id'], roles_ids)
        self.assertNotIn(role_foo_domain1['id'], roles_ids)
        self.assertNotIn(role_group_domain1['id'], roles_ids)

    def test_remote_user_no_realm(self):
        app = self.loadapp()
        auth_contexts = []

        def new_init(self, *args, **kwargs):
            super(auth.core.AuthContext, self).__init__(*args, **kwargs)
            auth_contexts.append(self)
        self.useFixture(fixtures.MockPatch('keystone.auth.core.AuthContext.__init__', new_init))
        with app.test_client() as c:
            c.environ_base.update(self.build_external_auth_environ(self.default_domain_user['name']))
            auth_req = self.build_authentication_request()
            c.post('/v3/auth/tokens', json=auth_req)
            self.assertEqual(self.default_domain_user['id'], auth_contexts[-1]['user_id'])
        user = {'name': 'myname@mydivision'}
        PROVIDERS.identity_api.update_user(self.default_domain_user['id'], user)
        with app.test_client() as c:
            c.environ_base.update(self.build_external_auth_environ(user['name']))
            auth_req = self.build_authentication_request()
            c.post('/v3/auth/tokens', json=auth_req)
            self.assertEqual(self.default_domain_user['id'], auth_contexts[-1]['user_id'])
        self.assertEqual(self.default_domain_user['id'], auth_contexts[-1]['user_id'])

    def test_remote_user_no_domain(self):
        app = self.loadapp()
        with app.test_client() as c:
            c.environ_base.update(self.build_external_auth_environ(self.user['name']))
            auth_request = self.build_authentication_request()
            c.post('/v3/auth/tokens', json=auth_request, expected_status_code=http.client.UNAUTHORIZED)

    def test_remote_user_and_password(self):
        app = self.loadapp()
        with app.test_client() as c:
            auth_data = self.build_authentication_request(user_domain_id=self.default_domain_user['domain_id'], username=self.default_domain_user['name'], password=self.default_domain_user['password'])
            c.post('/v3/auth/tokens', json=auth_data)

    def test_remote_user_and_explicit_external(self):
        auth_data = self.build_authentication_request(user_domain_id=self.domain['id'], username=self.user['name'], password=self.user['password'])
        auth_data['auth']['identity']['methods'] = ['password', 'external']
        auth_data['auth']['identity']['external'] = {}
        app = self.loadapp()
        with app.test_client() as c:
            c.post('/v3/auth/tokens', json=auth_data, expected_status_code=http.client.UNAUTHORIZED)

    def test_remote_user_bad_password(self):
        app = self.loadapp()
        auth_data = self.build_authentication_request(user_domain_id=self.domain['id'], username=self.user['name'], password='badpassword')
        with app.test_client() as c:
            c.post('/v3/auth/tokens', json=auth_data, expected_status_code=http.client.UNAUTHORIZED)

    def test_fetch_expired_allow_expired(self):
        self.config_fixture.config(group='token', expiration=10, allow_expired_window=20)
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            token = self._get_project_scoped_token()
            frozen_datetime.tick(delta=datetime.timedelta(seconds=2))
            self._validate_token(token)
            frozen_datetime.tick(delta=datetime.timedelta(seconds=12))
            self._validate_token(token, expected_status=http.client.NOT_FOUND)
            self._validate_token(token, allow_expired=True)
            frozen_datetime.tick(delta=datetime.timedelta(seconds=22))
            self._validate_token(token, allow_expired=True, expected_status=http.client.NOT_FOUND)

    def test_system_scoped_token_works_with_domain_specific_drivers(self):
        self.config_fixture.config(group='identity', domain_specific_drivers_enabled=True)
        PROVIDERS.assignment_api.create_system_grant_for_user(self.user['id'], self.role['id'])
        token_id = self.get_system_scoped_token()
        headers = {'X-Auth-Token': token_id}
        app = self.loadapp()
        with app.test_client() as c:
            c.get('/v3/users', headers=headers)

    def test_fetch_expired_allow_expired_in_expired_window(self):
        self.config_fixture.config(group='token', expiration=10, allow_expired_window=20)
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time):
            token = self._get_project_scoped_token()
        tick = datetime.timedelta(seconds=15)
        with freezegun.freeze_time(time + tick):
            self._validate_token(token, expected_status=http.client.NOT_FOUND)
            r = self._validate_token(token, allow_expired=True)
            self.assertValidProjectScopedTokenResponse(r)

    def _create_project_user(self):
        new_domain_ref = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(new_domain_ref['id'], new_domain_ref)
        new_project_ref = unit.new_project_ref(domain_id=self.domain_id)
        PROVIDERS.resource_api.create_project(new_project_ref['id'], new_project_ref)
        new_user = unit.create_user(PROVIDERS.identity_api, domain_id=new_domain_ref['id'], project_id=new_project_ref['id'])
        PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=new_user['id'], project_id=new_project_ref['id'])
        return (new_user, new_domain_ref, new_project_ref)

    def _create_certificates(self, root_dn=None, server_dn=None, client_dn=None):
        root_subj = unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organization_name='fujitsu', organizational_unit_name='test', common_name='root')
        if root_dn:
            root_subj = unit.update_dn(root_subj, root_dn)
        root_cert, root_key = unit.create_certificate(root_subj)
        keystone_subj = unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organization_name='fujitsu', organizational_unit_name='test', common_name='keystone.local')
        if server_dn:
            keystone_subj = unit.update_dn(keystone_subj, server_dn)
        ks_cert, ks_key = unit.create_certificate(keystone_subj, ca=root_cert, ca_key=root_key)
        client_subj = unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organization_name='fujitsu', organizational_unit_name='test', common_name='client')
        if client_dn:
            client_subj = unit.update_dn(client_subj, client_dn)
        client_cert, client_key = unit.create_certificate(client_subj, ca=root_cert, ca_key=root_key)
        return (root_cert, root_key, ks_cert, ks_key, client_cert, client_key)

    def _get_cert_content(self, cert):
        return cert.public_bytes(Encoding.PEM).decode('ascii')

    def _get_oauth2_access_token(self, client_id, client_cert_content, expected_status=http.client.OK):
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {'grant_type': 'client_credentials', 'client_id': client_id}
        extra_environ = {'SSL_CLIENT_CERT': client_cert_content}
        data = parse.urlencode(data).encode()
        resp = self.post('/OS-OAUTH2/token', headers=headers, noauth=True, convert=False, body=data, environ=extra_environ, expected_status=expected_status)
        return resp

    def _create_mapping(self):
        mapping = {'id': 'oauth2_mapping', 'rules': [{'local': [{'user': {'name': '{0}', 'id': '{1}', 'email': '{2}', 'domain': {'name': '{3}', 'id': '{4}'}}}], 'remote': [{'type': 'SSL_CLIENT_SUBJECT_DN_CN'}, {'type': 'SSL_CLIENT_SUBJECT_DN_UID'}, {'type': 'SSL_CLIENT_SUBJECT_DN_EMAILADDRESS'}, {'type': 'SSL_CLIENT_SUBJECT_DN_O'}, {'type': 'SSL_CLIENT_SUBJECT_DN_DC'}, {'type': 'SSL_CLIENT_ISSUER_DN_CN', 'any_one_of': ['root']}]}]}
        PROVIDERS.federation_api.create_mapping(mapping['id'], mapping)

    def test_verify_oauth2_token_project_scope_ok(self):
        cache_on_issue = CONF.token.cache_on_issue
        caching = CONF.token.caching
        self._create_mapping()
        user, user_domain, _ = self._create_project_user()
        *_, client_cert, _ = self._create_certificates(root_dn=unit.create_dn(common_name='root'), client_dn=unit.create_dn(common_name=user['name'], user_id=user['id'], email_address=user['email'], organization_name=user_domain['name'], domain_component=user_domain['id']))
        cert_content = self._get_cert_content(client_cert)
        CONF.token.cache_on_issue = False
        CONF.token.caching = False
        resp = self._get_oauth2_access_token(user['id'], cert_content)
        json_resp = json.loads(resp.body)
        self.assertIn('access_token', json_resp)
        self.assertEqual('Bearer', json_resp['token_type'])
        self.assertEqual(3600, json_resp['expires_in'])
        verify_resp = self.get('/auth/tokens', headers={'X-Subject-Token': json_resp['access_token'], 'X-Auth-Token': json_resp['access_token']}, expected_status=http.client.OK)
        self.assertIn('token', verify_resp.result)
        self.assertIn('oauth2_credential', verify_resp.result['token'])
        self.assertIn('roles', verify_resp.result['token'])
        self.assertIn('project', verify_resp.result['token'])
        self.assertIn('catalog', verify_resp.result['token'])
        check_oauth2 = verify_resp.result['token']['oauth2_credential']
        self.assertEqual(utils.get_certificate_thumbprint(cert_content), check_oauth2['x5t#S256'])
        CONF.token.cache_on_issue = cache_on_issue
        CONF.token.caching = caching