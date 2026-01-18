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
class DelayedAuthTests(BaseAuthTokenMiddlewareTest):

    def token_response(self, request, context):
        auth_id = request.headers.get('X-Auth-Token')
        self.assertEqual(auth_id, FAKE_ADMIN_TOKEN_ID)
        if request.headers.get('X-Subject-Token') == ERROR_TOKEN:
            msg = 'Network connection refused.'
            raise ksa_exceptions.ConnectFailure(msg)
        context.status_code = 404
        return ''

    def test_header_in_401(self):
        body = uuid.uuid4().hex
        www_authenticate_uri = 'http://local.test'
        conf = {'delay_auth_decision': 'True', 'auth_version': 'v3', 'www_authenticate_uri': www_authenticate_uri}
        middleware = self.create_simple_middleware(status='401 Unauthorized', body=body, conf=conf)
        resp = self.call(middleware, expected_status=401)
        self.assertEqual(body.encode(), resp.body)
        self.assertEqual('Keystone uri="%s"' % www_authenticate_uri, resp.headers['WWW-Authenticate'])

    def test_delayed_auth_values(self):
        conf = {'www_authenticate_uri': 'http://local.test'}
        status = '401 Unauthorized'
        middleware = self.create_simple_middleware(status=status, conf=conf)
        self.assertFalse(middleware._delay_auth_decision)
        for v in ('True', '1', 'on', 'yes'):
            conf = {'delay_auth_decision': v, 'www_authenticate_uri': 'http://local.test'}
            middleware = self.create_simple_middleware(status=status, conf=conf)
            self.assertTrue(middleware._delay_auth_decision)
        for v in ('False', '0', 'no'):
            conf = {'delay_auth_decision': v, 'www_authenticate_uri': 'http://local.test'}
            middleware = self.create_simple_middleware(status=status, conf=conf)
            self.assertFalse(middleware._delay_auth_decision)

    def test_auth_plugin_with_no_tokens(self):
        body = uuid.uuid4().hex
        www_authenticate_uri = 'http://local.test'
        conf = {'delay_auth_decision': True, 'www_authenticate_uri': www_authenticate_uri}
        middleware = self.create_simple_middleware(body=body, conf=conf)
        resp = self.call(middleware)
        self.assertEqual(body.encode(), resp.body)
        token_auth = resp.request.environ['keystone.token_auth']
        self.assertFalse(token_auth.has_user_token)
        self.assertIsNone(token_auth.user)
        self.assertFalse(token_auth.has_service_token)
        self.assertIsNone(token_auth.service)

    def test_auth_plugin_with_token(self):
        self.requests_mock.get('%s/v3/auth/tokens' % BASE_URI, text=self.token_response, headers={'X-Subject-Token': uuid.uuid4().hex})
        body = uuid.uuid4().hex
        www_authenticate_uri = 'http://local.test'
        conf = {'delay_auth_decision': 'True', 'www_authenticate_uri': www_authenticate_uri, 'auth_type': 'admin_token', 'endpoint': '%s/v3' % BASE_URI, 'token': FAKE_ADMIN_TOKEN_ID}
        middleware = self.create_simple_middleware(body=body, conf=conf)
        resp = self.call(middleware, headers={'X-Auth-Token': 'non-keystone-token'})
        self.assertEqual(body.encode(), resp.body)
        token_auth = resp.request.environ['keystone.token_auth']
        self.assertFalse(token_auth.has_user_token)
        self.assertIsNone(token_auth.user)
        self.assertFalse(token_auth.has_service_token)
        self.assertIsNone(token_auth.service)

    def test_auth_plugin_with_token_keystone_down(self):
        self.requests_mock.get('%s/v3/auth/tokens' % BASE_URI, text=self.token_response, headers={'X-Subject-Token': ERROR_TOKEN})
        body = uuid.uuid4().hex
        www_authenticate_uri = 'http://local.test'
        conf = {'delay_auth_decision': 'True', 'www_authenticate_uri': www_authenticate_uri, 'auth_type': 'admin_token', 'endpoint': '%s/v3' % BASE_URI, 'token': FAKE_ADMIN_TOKEN_ID, 'http_request_max_retries': '0'}
        middleware = self.create_simple_middleware(body=body, conf=conf)
        resp = self.call(middleware, headers={'X-Auth-Token': ERROR_TOKEN})
        self.assertEqual(body.encode(), resp.body)
        token_auth = resp.request.environ['keystone.token_auth']
        self.assertFalse(token_auth.has_user_token)
        self.assertIsNone(token_auth.user)
        self.assertFalse(token_auth.has_service_token)
        self.assertIsNone(token_auth.service)