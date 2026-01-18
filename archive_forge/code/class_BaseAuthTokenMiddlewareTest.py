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
class BaseAuthTokenMiddlewareTest(base.BaseAuthTokenTestCase):
    """Base test class for auth_token middleware.

    All the tests allow for running with auth_token
    configured for receiving v2 or v3 tokens, with the
    choice being made by passing configuration data into
    setUp().

    The base class will, by default, run all the tests
    expecting v2 token formats.  Child classes can override
    this to specify, for instance, v3 format.

    """

    def setUp(self, expected_env=None, auth_version=None, fake_app=None):
        super(BaseAuthTokenMiddlewareTest, self).setUp()
        self.logger = self.useFixture(fixtures.FakeLogger(level=logging.DEBUG))
        self.useFixture(fixtures.MockPatchObject(auth_token.AuthProtocol, '_create_oslo_cache', return_value=FakeOsloCache()))
        self.expected_env = expected_env or dict()
        self.fake_app = fake_app or FakeApp
        self.middleware = None
        self.conf = {'identity_uri': 'https://keystone.example.com:1234/testadmin/', 'auth_version': auth_version, 'www_authenticate_uri': 'https://keystone.example.com:1234', 'admin_user': uuid.uuid4().hex}
        self.auth_version = auth_version
        self.response_status = None
        self.response_headers = None

    def call_middleware(self, **kwargs):
        return self.call(self.middleware, **kwargs)

    def set_middleware(self, expected_env=None, conf=None):
        """Configure the class ready to call the auth_token middleware.

        Set up the various fake items needed to run the middleware.
        Individual tests that need to further refine these can call this
        function to override the class defaults.

        """
        if conf:
            self.conf.update(conf)
        if expected_env:
            self.expected_env.update(expected_env)
        self.middleware = auth_token.AuthProtocol(self.fake_app(self.expected_env), self.conf)

    def update_expected_env(self, expected_env={}):
        self.middleware._app.expected_env.update(expected_env)

    def purge_token_expected_env(self):
        for key in self.token_expected_env.keys():
            del self.middleware._app.expected_env[key]

    def purge_service_token_expected_env(self):
        for key in self.service_token_expected_env.keys():
            del self.middleware._app.expected_env[key]

    def assertLastPath(self, path):
        if path:
            self.assertEqual(BASE_URI + path, self.requests_mock.last_request.url)
        else:
            self.assertIsNone(self.requests_mock.last_request)