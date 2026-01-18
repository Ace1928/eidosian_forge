import datetime
import os
import time
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystoneauth1 import session
import oslo_cache
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import pbr.version
import testresources
from testtools import matchers
import webob
import webob.dec
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _base
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import client_fixtures
class v3AuthTokenMiddlewareTest(BaseAuthTokenMiddlewareTest, CommonAuthTokenMiddlewareTest, testresources.ResourcedTestCase):
    """Test auth_token middleware with v3 tokens.

    Re-execute the AuthTokenMiddlewareTest class tests, but with the
    auth_token middleware configured to expect v3 tokens back from
    a keystone server.

    This is done by configuring the AuthTokenMiddlewareTest class via
    its Setup(), passing in v3 style data that will then be used by
    the tests themselves.

    """
    resources = [('examples', client_fixtures.EXAMPLES_RESOURCE)]

    def setUp(self):
        super(v3AuthTokenMiddlewareTest, self).setUp(auth_version='v3.0', fake_app=v3FakeApp)
        self.token_dict = {'uuid_token_default': self.examples.v3_UUID_TOKEN_DEFAULT, 'uuid_token_unscoped': self.examples.v3_UUID_TOKEN_UNSCOPED, 'uuid_token_bind': self.examples.v3_UUID_TOKEN_BIND, 'uuid_token_unknown_bind': self.examples.v3_UUID_TOKEN_UNKNOWN_BIND, 'uuid_service_token_default': self.examples.v3_UUID_SERVICE_TOKEN_DEFAULT}
        self.requests_mock.get(BASE_URI, json=VERSION_LIST_v3, status_code=300)
        self.requests_mock.post('%s/v2.0/tokens' % BASE_URI, text=FAKE_ADMIN_TOKEN)
        self.requests_mock.get('%s/v3/auth/tokens' % BASE_URI, text=self.token_response, headers={'X-Subject-Token': uuid.uuid4().hex})
        self.set_middleware()

    def token_response(self, request, context):
        auth_id = request.headers.get('X-Auth-Token')
        token_id = request.headers.get('X-Subject-Token')
        self.assertEqual(auth_id, FAKE_ADMIN_TOKEN_ID)
        if token_id == ERROR_TOKEN:
            msg = 'Network connection refused.'
            raise ksa_exceptions.ConnectFailure(msg)
        elif token_id == TIMEOUT_TOKEN:
            request_timeout_response(request, context)
        elif token_id == ENDPOINT_NOT_FOUND_TOKEN:
            raise ksa_exceptions.EndpointNotFound()
        try:
            response = self.examples.JSON_TOKEN_RESPONSES[token_id]
        except KeyError:
            response = ''
            context.status_code = 404
        return response

    def assert_valid_last_url(self, token_id):
        self.assertLastPath('/v3/auth/tokens')

    def test_valid_unscoped_uuid_request(self):
        delta_expected_env = {'HTTP_X_PROJECT_ID': None, 'HTTP_X_PROJECT_NAME': None, 'HTTP_X_PROJECT_DOMAIN_ID': None, 'HTTP_X_PROJECT_DOMAIN_NAME': None, 'HTTP_X_TENANT_ID': None, 'HTTP_X_TENANT_NAME': None, 'HTTP_X_ROLES': '', 'HTTP_X_TENANT': None, 'HTTP_X_ROLE': ''}
        self.set_middleware(expected_env=delta_expected_env)
        self.assert_valid_request_200(self.examples.v3_UUID_TOKEN_UNSCOPED, with_catalog=False)
        self.assertLastPath('/v3/auth/tokens')

    def test_valid_system_scoped_token_request(self):
        delta_expected_env = {'HTTP_OPENSTACK_SYSTEM_SCOPE': 'all', 'HTTP_X_PROJECT_ID': None, 'HTTP_X_PROJECT_NAME': None, 'HTTP_X_PROJECT_DOMAIN_ID': None, 'HTTP_X_PROJECT_DOMAIN_NAME': None, 'HTTP_X_TENANT_ID': None, 'HTTP_X_TENANT_NAME': None, 'HTTP_X_TENANT': None}
        self.set_middleware(expected_env=delta_expected_env)
        self.assert_valid_request_200(self.examples.v3_SYSTEM_SCOPED_TOKEN)
        self.assertLastPath('/v3/auth/tokens')

    def test_domain_scoped_uuid_request(self):
        delta_expected_env = {'HTTP_X_DOMAIN_ID': 'domain_id1', 'HTTP_X_DOMAIN_NAME': 'domain_name1', 'HTTP_X_PROJECT_ID': None, 'HTTP_X_PROJECT_NAME': None, 'HTTP_X_PROJECT_DOMAIN_ID': None, 'HTTP_X_PROJECT_DOMAIN_NAME': None, 'HTTP_X_TENANT_ID': None, 'HTTP_X_TENANT_NAME': None, 'HTTP_X_TENANT': None}
        self.set_middleware(expected_env=delta_expected_env)
        self.assert_valid_request_200(self.examples.v3_UUID_TOKEN_DOMAIN_SCOPED)
        self.assertLastPath('/v3/auth/tokens')

    def test_user_plugin_token_properties(self):
        token = self.examples.v3_UUID_TOKEN_DEFAULT
        token_data = self.examples.TOKEN_RESPONSES[token]
        service = self.examples.v3_UUID_SERVICE_TOKEN_DEFAULT
        service_data = self.examples.TOKEN_RESPONSES[service]
        resp = self.call_middleware(headers={'X-Service-Catalog': '[]', 'X-Auth-Token': token, 'X-Service-Token': service})
        self.assertEqual(FakeApp.SUCCESS, resp.body)
        token_auth = resp.request.environ['keystone.token_auth']
        self.assertTrue(token_auth.has_user_token)
        self.assertTrue(token_auth.has_service_token)
        self.assertEqual(token_data.user_id, token_auth.user.user_id)
        self.assertEqual(token_data.project_id, token_auth.user.project_id)
        self.assertEqual(token_data.user_domain_id, token_auth.user.user_domain_id)
        self.assertEqual(token_data.project_domain_id, token_auth.user.project_domain_id)
        self.assertThat(token_auth.user.role_names, matchers.HasLength(2))
        self.assertIn('role1', token_auth.user.role_names)
        self.assertIn('role2', token_auth.user.role_names)
        self.assertIsNone(token_auth.user.trust_id)
        self.assertEqual(service_data.user_id, token_auth.service.user_id)
        self.assertEqual(service_data.project_id, token_auth.service.project_id)
        self.assertEqual(service_data.user_domain_id, token_auth.service.user_domain_id)
        self.assertEqual(service_data.project_domain_id, token_auth.service.project_domain_id)
        self.assertThat(token_auth.service.role_names, matchers.HasLength(2))
        self.assertIn('service', token_auth.service.role_names)
        self.assertIn('service_role2', token_auth.service.role_names)
        self.assertIsNone(token_auth.service.trust_id)

    def test_expire_stored_in_cache(self):
        token = 'mytoken'
        data = 'this_data'
        self.set_middleware()
        self.middleware._token_cache.initialize({})
        now = datetime.datetime.now(datetime.timezone.utc)
        delta = datetime.timedelta(hours=1)
        expires = strtime(at=now + delta)
        self.middleware._token_cache.set(token, (data, expires))
        new_data = self.middleware.fetch_token(token)
        self.assertEqual(data, new_data)

    def test_endpoint_not_found_in_token(self):
        token = ENDPOINT_NOT_FOUND_TOKEN
        self.set_middleware()
        self.middleware._token_cache.initialize({})
        with mock.patch.object(self.middleware._identity_server, 'invalidate', new=mock.Mock()):
            self.assertRaises(ksa_exceptions.EndpointNotFound, self.middleware.fetch_token, token)
            self.assertTrue(self.middleware._identity_server.invalidate.called)

    def test_not_is_admin_project(self):
        token = self.examples.v3_NOT_IS_ADMIN_PROJECT
        self.set_middleware(expected_env={'HTTP_X_IS_ADMIN_PROJECT': 'False'})
        req = self.assert_valid_request_200(token)
        self.assertIs(False, req.environ['keystone.token_auth'].user.is_admin_project)

    def test_service_token_with_valid_service_role_not_required(self):
        s = super(v3AuthTokenMiddlewareTest, self)
        s.test_service_token_with_valid_service_role_not_required()
        e = self.requests_mock.request_history[3].qs.get('allow_expired')
        self.assertEqual(['1'], e)

    def test_service_token_with_invalid_service_role_not_required(self):
        s = super(v3AuthTokenMiddlewareTest, self)
        s.test_service_token_with_invalid_service_role_not_required()
        e = self.requests_mock.request_history[3].qs.get('allow_expired')
        self.assertIsNone(e)

    def test_service_token_with_valid_service_role_required(self):
        s = super(v3AuthTokenMiddlewareTest, self)
        s.test_service_token_with_valid_service_role_required()
        e = self.requests_mock.request_history[3].qs.get('allow_expired')
        self.assertEqual(['1'], e)

    def test_service_token_with_invalid_service_role_required(self):
        s = super(v3AuthTokenMiddlewareTest, self)
        s.test_service_token_with_invalid_service_role_required()
        e = self.requests_mock.request_history[3].qs.get('allow_expired')
        self.assertIsNone(e)

    def test_app_cred_token_without_access_rules(self):
        self.set_middleware(conf={'service_type': 'compute'})
        token = self.examples.v3_APP_CRED_TOKEN
        token_data = self.examples.TOKEN_RESPONSES[token]
        resp = self.call_middleware(headers={'X-Auth-Token': token})
        self.assertEqual(FakeApp.SUCCESS, resp.body)
        token_auth = resp.request.environ['keystone.token_auth']
        self.assertEqual(token_data.application_credential_id, token_auth.user.application_credential_id)

    def test_app_cred_access_rules_token(self):
        self.set_middleware(conf={'service_type': 'compute'})
        token = self.examples.v3_APP_CRED_ACCESS_RULES
        token_data = self.examples.TOKEN_RESPONSES[token]
        resp = self.call_middleware(headers={'X-Auth-Token': token}, expected_status=200, method='GET', path='/v2.1/servers')
        token_auth = resp.request.environ['keystone.token_auth']
        self.assertEqual(token_data.application_credential_id, token_auth.user.application_credential_id)
        self.assertEqual(token_data.application_credential_access_rules, token_auth.user.application_credential_access_rules)
        resp = self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/v2.1/servers/someuuid')
        token_auth = resp.request.environ['keystone.token_auth']
        self.assertEqual(token_data.application_credential_id, token_auth.user.application_credential_id)
        self.assertEqual(token_data.application_credential_access_rules, token_auth.user.application_credential_access_rules)

    def test_app_cred_access_rules_service_request(self):
        self.set_middleware(conf={'service_type': 'image'})
        token = self.examples.v3_APP_CRED_ACCESS_RULES
        headers = {'X-Auth-Token': token}
        self.call_middleware(headers=headers, expected_status=401, method='GET', path='/v2/images')
        service_token = self.examples.v3_UUID_SERVICE_TOKEN_DEFAULT
        headers['X-Service-Token'] = service_token
        self.call_middleware(headers=headers, expected_status=200, method='GET', path='/v2/images')

    def test_app_cred_no_access_rules_token(self):
        self.set_middleware(conf={'service_type': 'compute'})
        token = self.examples.v3_APP_CRED_EMPTY_ACCESS_RULES
        self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/v2.1/servers')
        service_token = self.examples.v3_UUID_SERVICE_TOKEN_DEFAULT
        headers = {'X-Auth-Token': token, 'X-Service-Token': service_token}
        self.call_middleware(headers=headers, expected_status=401, method='GET', path='/v2.1/servers')

    def test_app_cred_matching_rules(self):
        self.set_middleware(conf={'service_type': 'compute'})
        token = self.examples.v3_APP_CRED_MATCHING_RULES
        self.call_middleware(headers={'X-Auth-Token': token}, expected_status=200, method='GET', path='/v2.1/servers/foobar')
        self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/v2.1/servers/foobar/barfoo')
        self.set_middleware(conf={'service_type': 'image'})
        self.call_middleware(headers={'X-Auth-Token': token}, expected_status=200, method='GET', path='/v2/images/foobar')
        self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/v2/images/foobar/barfoo')
        self.set_middleware(conf={'service_type': 'identity'})
        self.call_middleware(headers={'X-Auth-Token': token}, expected_status=200, method='GET', path='/v3/projects/123/users/456/roles/member')
        self.set_middleware(conf={'service_type': 'block-storage'})
        self.call_middleware(headers={'X-Auth-Token': token}, expected_status=200, method='GET', path='/v3/123/types/456')
        self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/v3/123/types')
        self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/v2/123/types/456')
        self.set_middleware(conf={'service_type': 'object-store'})
        self.call_middleware(headers={'X-Auth-Token': token}, expected_status=200, method='GET', path='/v1/1/2/3')
        self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/v1/1/2')
        self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/v2/1/2')
        self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/info')