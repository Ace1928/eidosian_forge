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
class CommonCompositeAuthTests(object):
    """Test Composite authentication.

    Test the behaviour of adding a service-token.
    """

    def test_composite_auth_ok(self):
        token = self.token_dict['uuid_token_default']
        service_token = self.token_dict['uuid_service_token_default']
        fake_logger = fixtures.FakeLogger(level=logging.DEBUG)
        self.middleware.logger = self.useFixture(fake_logger)
        resp = self.call_middleware(headers={'X-Auth-Token': token, 'X-Service-Token': service_token})
        self.assertEqual(FakeApp.SUCCESS, resp.body)
        expected_env = dict(EXPECTED_V2_DEFAULT_ENV_RESPONSE)
        expected_env.update(EXPECTED_V2_DEFAULT_SERVICE_ENV_RESPONSE)
        self.assertIn('Received request from user: ', fake_logger.output)
        self.assertIn('user_id %(HTTP_X_USER_ID)s, project_id %(HTTP_X_TENANT_ID)s, roles ' % expected_env, fake_logger.output)
        self.assertIn('service: user_id %(HTTP_X_SERVICE_USER_ID)s, project_id %(HTTP_X_SERVICE_PROJECT_ID)s, roles ' % expected_env, fake_logger.output)
        roles = ','.join([expected_env['HTTP_X_SERVICE_ROLES'], expected_env['HTTP_X_ROLES']])
        for r in roles.split(','):
            self.assertIn(r, fake_logger.output)

    def test_composite_auth_invalid_service_token(self):
        token = self.token_dict['uuid_token_default']
        service_token = 'invalid-service-token'
        resp = self.call_middleware(headers={'X-Auth-Token': token, 'X-Service-Token': service_token}, expected_status=401)
        expected_body = b'The request you have made requires authentication.'
        self.assertThat(resp.body, matchers.Contains(expected_body))

    def test_composite_auth_no_service_token(self):
        self.purge_service_token_expected_env()
        req = webob.Request.blank('/')
        req.headers['X-Auth-Token'] = self.token_dict['uuid_token_default']
        for key, value in self.service_token_expected_env.items():
            header_key = key[len('HTTP_'):].replace('_', '-')
            req.headers[header_key] = value
        req.headers['X-Foo'] = 'Bar'
        resp = req.get_response(self.middleware)
        for key in self.service_token_expected_env.keys():
            header_key = key[len('HTTP_'):].replace('_', '-')
            self.assertFalse(req.headers.get(header_key))
        self.assertEqual('Bar', req.headers.get('X-Foo'))
        self.assertEqual(418, resp.status_int)
        self.assertEqual(FakeApp.FORBIDDEN, resp.body)

    def test_composite_auth_invalid_user_token(self):
        token = 'invalid-token'
        service_token = self.token_dict['uuid_service_token_default']
        resp = self.call_middleware(headers={'X-Auth-Token': token, 'X-Service-Token': service_token}, expected_status=401)
        expected_body = b'The request you have made requires authentication.'
        self.assertThat(resp.body, matchers.Contains(expected_body))

    def test_composite_auth_no_user_token(self):
        service_token = self.token_dict['uuid_service_token_default']
        resp = self.call_middleware(headers={'X-Service-Token': service_token}, expected_status=401)
        expected_body = b'The request you have made requires authentication.'
        self.assertThat(resp.body, matchers.Contains(expected_body))

    def test_composite_auth_delay_ok(self):
        self.middleware._delay_auth_decision = True
        token = self.token_dict['uuid_token_default']
        service_token = self.token_dict['uuid_service_token_default']
        resp = self.call_middleware(headers={'X-Auth-Token': token, 'X-Service-Token': service_token})
        self.assertEqual(FakeApp.SUCCESS, resp.body)

    def test_composite_auth_delay_invalid_service_token(self):
        self.middleware._delay_auth_decision = True
        self.purge_service_token_expected_env()
        expected_env = {'HTTP_X_SERVICE_IDENTITY_STATUS': 'Invalid'}
        self.update_expected_env(expected_env)
        token = self.token_dict['uuid_token_default']
        service_token = 'invalid-service-token'
        resp = self.call_middleware(headers={'X-Auth-Token': token, 'X-Service-Token': service_token}, expected_status=420)
        self.assertEqual(FakeApp.FORBIDDEN, resp.body)

    def test_composite_auth_delay_invalid_service_and_user_tokens(self):
        self.middleware._delay_auth_decision = True
        self.purge_service_token_expected_env()
        self.purge_token_expected_env()
        expected_env = {'HTTP_X_IDENTITY_STATUS': 'Invalid', 'HTTP_X_SERVICE_IDENTITY_STATUS': 'Invalid'}
        self.update_expected_env(expected_env)
        token = 'invalid-token'
        service_token = 'invalid-service-token'
        resp = self.call_middleware(headers={'X-Auth-Token': token, 'X-Service-Token': service_token}, expected_status=419)
        self.assertEqual(FakeApp.FORBIDDEN, resp.body)

    def test_composite_auth_delay_no_service_token(self):
        self.middleware._delay_auth_decision = True
        self.purge_service_token_expected_env()
        req = webob.Request.blank('/')
        req.headers['X-Auth-Token'] = self.token_dict['uuid_token_default']
        for key, value in self.service_token_expected_env.items():
            header_key = key[len('HTTP_'):].replace('_', '-')
            req.headers[header_key] = value
        req.headers['X-Foo'] = 'Bar'
        resp = req.get_response(self.middleware)
        for key in self.service_token_expected_env.keys():
            header_key = key[len('HTTP_'):].replace('_', '-')
            self.assertFalse(req.headers.get(header_key))
        self.assertEqual('Bar', req.headers.get('X-Foo'))
        self.assertEqual(418, resp.status_int)
        self.assertEqual(FakeApp.FORBIDDEN, resp.body)

    def test_composite_auth_delay_invalid_user_token(self):
        self.middleware._delay_auth_decision = True
        self.purge_token_expected_env()
        expected_env = {'HTTP_X_IDENTITY_STATUS': 'Invalid'}
        self.update_expected_env(expected_env)
        token = 'invalid-token'
        service_token = self.token_dict['uuid_service_token_default']
        resp = self.call_middleware(headers={'X-Auth-Token': token, 'X-Service-Token': service_token}, expected_status=403)
        self.assertEqual(FakeApp.FORBIDDEN, resp.body)

    def test_composite_auth_delay_no_user_token(self):
        self.middleware._delay_auth_decision = True
        self.purge_token_expected_env()
        expected_env = {'HTTP_X_IDENTITY_STATUS': 'Invalid'}
        self.update_expected_env(expected_env)
        service_token = self.token_dict['uuid_service_token_default']
        resp = self.call_middleware(headers={'X-Service-Token': service_token}, expected_status=403)
        self.assertEqual(FakeApp.FORBIDDEN, resp.body)

    def assert_kerberos_composite_bind(self, user_token, service_token, bind_level):
        conf = {'enforce_token_bind': bind_level, 'auth_version': self.auth_version}
        self.set_middleware(conf=conf)
        req = webob.Request.blank('/')
        req.headers['X-Auth-Token'] = user_token
        req.headers['X-Service-Token'] = service_token
        req.environ['REMOTE_USER'] = self.examples.SERVICE_KERBEROS_BIND
        req.environ['AUTH_TYPE'] = 'Negotiate'
        resp = req.get_response(self.middleware)
        self.assertEqual(200, resp.status_int)
        self.assertEqual(FakeApp.SUCCESS, resp.body)
        self.assertIn('keystone.token_info', req.environ)

    def test_composite_auth_with_bind(self):
        token = self.token_dict['uuid_token_bind']
        service_token = self.token_dict['uuid_service_token_bind']
        self.assert_kerberos_composite_bind(token, service_token, bind_level='required')