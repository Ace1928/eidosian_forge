import fixtures
import http.client as http_client
import logging
import testresources
import uuid
import webob.dec
from oslo_config import cfg
from keystoneauth1 import exceptions as ksa_exceptions
from keystonemiddleware import oauth2_token
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware\
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit import client_fixtures
from keystonemiddleware.tests.unit import utils
class Oauth2TokenMiddlewareTest(BaseOauth2TokenMiddlewareTest, testresources.ResourcedTestCase):
    resources = [('examples', client_fixtures.EXAMPLES_RESOURCE)]

    def setUp(self):
        super(Oauth2TokenMiddlewareTest, self).setUp(auth_version='v3.0', fake_app=FakeOauth2TokenV3App)
        self.requests_mock.post('%s/v2.0/tokens' % BASE_URI, text=FAKE_ADMIN_TOKEN)
        self.requests_mock.get(BASE_URI, json=VERSION_LIST_v3, status_code=300)
        self.requests_mock.get('%s/v3/auth/tokens' % BASE_URI, text=self.token_response, headers={'X-Subject-Token': uuid.uuid4().hex})
        self.set_middleware()

    def token_response(self, request, context):
        auth_id = request.headers.get('X-Auth-Token')
        token_id = request.headers.get('X-Subject-Token')
        self.assertEqual(auth_id, FAKE_ADMIN_TOKEN_ID)
        if token_id == ERROR_TOKEN:
            msg = 'Network connection refused.'
            raise ksa_exceptions.ConnectFailure(msg)
        if token_id == ENDPOINT_NOT_FOUND_TOKEN:
            raise ksa_exceptions.EndpointNotFound()
        if token_id == TIMEOUT_TOKEN:
            request_timeout_response(request, context)
        try:
            response = self.examples.JSON_TOKEN_RESPONSES[token_id]
        except KeyError:
            response = ''
            context.status_code = 404
        return response

    def test_app_cred_token_without_access_rules(self):
        self.set_middleware(conf={'service_type': 'compute'})
        token = self.examples.v3_APP_CRED_TOKEN
        token_data = self.examples.TOKEN_RESPONSES[token]
        resp = self.call_middleware(headers=get_authorization_header(token))
        self.assertEqual(FakeApp.SUCCESS, resp.body)
        token_auth = resp.request.environ['keystone.token_auth']
        self.assertEqual(token_data.application_credential_id, token_auth.user.application_credential_id)

    def test_app_cred_access_rules_token(self):
        self.set_middleware(conf={'service_type': 'compute'})
        token = self.examples.v3_APP_CRED_ACCESS_RULES
        token_data = self.examples.TOKEN_RESPONSES[token]
        resp = self.call_middleware(headers=get_authorization_header(token), expected_status=200, method='GET', path='/v2.1/servers')
        token_auth = resp.request.environ['keystone.token_auth']
        self.assertEqual(token_data.application_credential_id, token_auth.user.application_credential_id)
        self.assertEqual(token_data.application_credential_access_rules, token_auth.user.application_credential_access_rules)
        resp = self.call_middleware(headers=get_authorization_header(token), expected_status=401, method='GET', path='/v2.1/servers/someuuid')
        self.assertEqual(token_data.application_credential_id, token_auth.user.application_credential_id)
        self.assertEqual(token_data.application_credential_access_rules, token_auth.user.application_credential_access_rules)

    def test_app_cred_no_access_rules_token(self):
        self.set_middleware(conf={'service_type': 'compute'})
        token = self.examples.v3_APP_CRED_EMPTY_ACCESS_RULES
        self.call_middleware(headers=get_authorization_header(token), expected_status=401, method='GET', path='/v2.1/servers')

    def test_app_cred_matching_rules(self):
        self.set_middleware(conf={'service_type': 'compute'})
        token = self.examples.v3_APP_CRED_MATCHING_RULES
        self.call_middleware(headers=get_authorization_header(token), expected_status=200, method='GET', path='/v2.1/servers/foobar')
        self.call_middleware(headers=get_authorization_header(token), expected_status=401, method='GET', path='/v2.1/servers/foobar/barfoo')
        self.set_middleware(conf={'service_type': 'image'})
        self.call_middleware(headers=get_authorization_header(token), expected_status=200, method='GET', path='/v2/images/foobar')
        self.call_middleware(headers=get_authorization_header(token), expected_status=401, method='GET', path='/v2/images/foobar/barfoo')
        self.set_middleware(conf={'service_type': 'identity'})
        self.call_middleware(headers=get_authorization_header(token), expected_status=200, method='GET', path='/v3/projects/123/users/456/roles/member')
        self.set_middleware(conf={'service_type': 'block-storage'})
        self.call_middleware(headers=get_authorization_header(token), expected_status=200, method='GET', path='/v3/123/types/456')
        self.call_middleware(headers=get_authorization_header(token), expected_status=401, method='GET', path='/v3/123/types')
        self.call_middleware(headers=get_authorization_header(token), expected_status=401, method='GET', path='/v2/123/types/456')
        self.set_middleware(conf={'service_type': 'object-store'})
        self.call_middleware(headers=get_authorization_header(token), expected_status=200, method='GET', path='/v1/1/2/3')
        self.call_middleware(headers=get_authorization_header(token), expected_status=401, method='GET', path='/v1/1/2')
        self.call_middleware(headers=get_authorization_header(token), expected_status=401, method='GET', path='/v2/1/2')
        self.call_middleware(headers=get_authorization_header(token), expected_status=401, method='GET', path='/info')

    def test_request_no_token(self):
        resp = self.call_middleware(expected_status=401)
        self.assertEqual('Keystone uri="https://keystone.example.com:1234"', resp.headers['WWW-Authenticate'])

    def test_request_blank_token(self):
        resp = self.call_middleware(headers=get_authorization_header(''), expected_status=401)
        self.assertEqual('Keystone uri="https://keystone.example.com:1234"', resp.headers['WWW-Authenticate'])

    def test_request_not_app_cred_token(self):
        self.call_middleware(headers=get_authorization_header(self.examples.v3_UUID_TOKEN_DEFAULT), expected_status=200)

    def _get_cached_token(self, token):
        return self.middleware._token_cache.get(token)

    def assert_valid_last_url(self, token_id):
        self.assertLastPath('/v3/auth/tokens')

    def assertLastPath(self, path):
        if path:
            self.assertEqual(BASE_URI + path, self.requests_mock.last_request.url)
        else:
            self.assertIsNone(self.requests_mock.last_request)

    def test_http_error_not_cached_token(self):
        """Test to don't cache token as invalid on network errors.

        We use UUID tokens since they are the easiest one to reach
        get_http_connection.
        """
        self.set_middleware(conf={'http_request_max_retries': '0'})
        self.call_middleware(headers=get_authorization_header(ERROR_TOKEN), expected_status=503)
        self.assertIsNone(self._get_cached_token(ERROR_TOKEN))
        self.assert_valid_last_url(ERROR_TOKEN)