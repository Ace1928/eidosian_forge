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
class OtherTests(BaseAuthTokenMiddlewareTest):

    def setUp(self):
        super(OtherTests, self).setUp()
        self.logger = self.useFixture(fixtures.FakeLogger())

    def test_unknown_server_versions(self):
        versions = fixture.DiscoveryList(v2=False, v3_id='v4', href=BASE_URI)
        self.set_middleware()
        self.requests_mock.get(BASE_URI, json=versions, status_code=300)
        self.call_middleware(headers={'X-Auth-Token': uuid.uuid4().hex}, expected_status=503)
        self.assertIn('versions [v3.0]', self.logger.output)

    def _assert_auth_version(self, conf_version, identity_server_version):
        self.set_middleware(conf={'auth_version': conf_version})
        identity_server = self.middleware._create_identity_server()
        self.assertEqual(identity_server_version, identity_server.auth_version)

    def test_micro_version(self):
        self._assert_auth_version('v3', (3, 0))
        self._assert_auth_version('v3.0', (3, 0))
        self._assert_auth_version('v3.1', (3, 0))
        self._assert_auth_version('v3.2', (3, 0))
        self._assert_auth_version('v3.9', (3, 0))
        self._assert_auth_version('v3.3.1', (3, 0))
        self._assert_auth_version('v3.3.5', (3, 0))

    def test_default_auth_version(self):
        self.requests_mock.get(BASE_URI, json=VERSION_LIST_v3, status_code=300)
        self._assert_auth_version(None, (3, 0))

    def test_unsupported_auth_version(self):
        self._assert_auth_version('v1', (3, 0))
        self._assert_auth_version('v10', (3, 0))