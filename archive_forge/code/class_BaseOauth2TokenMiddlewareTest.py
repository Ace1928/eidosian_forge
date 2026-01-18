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
class BaseOauth2TokenMiddlewareTest(base.BaseAuthTokenTestCase):

    def setUp(self, expected_env=None, auth_version=None, fake_app=None):
        cfg.CONF.clear()
        super(BaseOauth2TokenMiddlewareTest, self).setUp()
        self.logger = self.useFixture(fixtures.FakeLogger(level=logging.DEBUG))
        self.useFixture(fixtures.MockPatchObject(oauth2_token.OAuth2Protocol, '_create_oslo_cache', return_value=FakeOsloCache))
        self.expected_env = expected_env or dict()
        self.fake_app = fake_app or FakeApp
        self.middleware = None
        self.conf = {'identity_uri': 'https://keystone.example.com:1234/testadmin/', 'auth_version': auth_version, 'www_authenticate_uri': 'https://keystone.example.com:1234', 'admin_user': uuid.uuid4().hex}
        self.auth_version = auth_version

    def call_middleware(self, **kwargs):
        return self.call(self.middleware, **kwargs)

    def set_middleware(self, expected_env=None, conf=None):
        """Configure the class ready to call the oauth2_token middleware.

        Set up the various fake items needed to run the middleware.
        Individual tests that need to further refine these can call this
        function to override the class defaults.

        """
        if conf:
            self.conf.update(conf)
        if expected_env:
            self.expected_env.update(expected_env)
        self.middleware = oauth2_token.OAuth2Protocol(self.fake_app(self.expected_env), self.conf)

    def call(self, middleware, method='GET', path='/', headers=None, expected_status=http_client.OK, expected_body_string=None):
        req = webob.Request.blank(path)
        req.method = method
        for k, v in (headers or {}).items():
            req.headers[k] = v
        resp = req.get_response(middleware)
        self.assertEqual(expected_status, resp.status_int)
        if expected_body_string:
            self.assertIn(expected_body_string, str(resp.body))
        resp.request = req
        return resp