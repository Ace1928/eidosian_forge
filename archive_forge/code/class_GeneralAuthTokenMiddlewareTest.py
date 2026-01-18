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
class GeneralAuthTokenMiddlewareTest(BaseAuthTokenMiddlewareTest, testresources.ResourcedTestCase):
    """General Token Behavior tests.

    These tests are not affected by the token format
    (see CommonAuthTokenMiddlewareTest).
    """
    resources = [('examples', client_fixtures.EXAMPLES_RESOURCE)]

    def test_fixed_cache_key_length(self):
        self.set_middleware()
        short_string = uuid.uuid4().hex
        long_string = 8 * uuid.uuid4().hex
        token_cache = self.middleware._token_cache
        hashed_short_string_key, context_ = token_cache._get_cache_key(short_string)
        hashed_long_string_key, context_ = token_cache._get_cache_key(long_string)
        self.assertThat(hashed_short_string_key, matchers.HasLength(len(hashed_long_string_key)))

    def test_conf_values_type_convert(self):
        conf = {'identity_uri': 'https://keystone.example.com:1234', 'include_service_catalog': '0', 'nonexsit_option': '0'}
        middleware = auth_token.AuthProtocol(self.fake_app, conf)
        self.assertFalse(middleware._include_service_catalog)
        self.assertEqual('0', middleware._conf.get('nonexsit_option'))

    def test_deprecated_conf_values(self):
        servers = 'localhost:11211'
        conf = {'memcache_servers': servers}
        middleware = auth_token.AuthProtocol(self.fake_app, conf)
        self.assertEqual([servers], middleware._conf.get('memcached_servers'))

    def test_conf_values_type_convert_with_wrong_key(self):
        conf = {'wrong_key': '123'}
        log = 'The option "wrong_key" is not known to keystonemiddleware'
        auth_token.AuthProtocol(self.fake_app, conf)
        self.assertThat(self.logger.output, matchers.Contains(log))

    def test_conf_values_type_convert_with_wrong_value(self):
        conf = {'include_service_catalog': '123'}
        self.assertRaises(ksm_exceptions.ConfigurationError, auth_token.AuthProtocol, self.fake_app, conf)

    def test_auth_region_name(self):
        token = fixture.V3Token()
        auth_url = 'http://keystone-auth.example.com:5000'
        east_url = 'http://keystone-east.example.com:5000'
        west_url = 'http://keystone-west.example.com:5000'
        auth_versions = fixture.DiscoveryList(href=auth_url)
        east_versions = fixture.DiscoveryList(href=east_url)
        west_versions = fixture.DiscoveryList(href=west_url)
        s = token.add_service('identity')
        s.add_endpoint(interface='internal', url=east_url, region='east')
        s.add_endpoint(interface='internal', url=west_url, region='west')
        self.requests_mock.get(auth_url, json=auth_versions)
        self.requests_mock.get(east_url, json=east_versions)
        self.requests_mock.get(west_url, json=west_versions)
        self.requests_mock.post('%s/v3/auth/tokens' % auth_url, headers={'X-Subject-Token': uuid.uuid4().hex}, json=token)
        east_mock = self.requests_mock.get('%s/v3/auth/tokens' % east_url, headers={'X-Subject-Token': uuid.uuid4().hex}, json=fixture.V3Token())
        west_mock = self.requests_mock.get('%s/v3/auth/tokens' % west_url, headers={'X-Subject-Token': uuid.uuid4().hex}, json=fixture.V3Token())
        loading.register_auth_conf_options(self.cfg.conf, group=_base.AUTHTOKEN_GROUP)
        opts = loading.get_auth_plugin_conf_options('v3password')
        self.cfg.register_opts(opts, group=_base.AUTHTOKEN_GROUP)
        self.cfg.config(auth_url=auth_url + '/v3', auth_type='v3password', username='user', password='pass', user_domain_id=uuid.uuid4().hex, group=_base.AUTHTOKEN_GROUP)
        self.assertEqual(0, east_mock.call_count)
        self.assertEqual(0, west_mock.call_count)
        east_app = self.create_simple_middleware(conf=dict(region_name='east'))
        self.call(east_app, headers={'X-Auth-Token': uuid.uuid4().hex})
        self.assertEqual(1, east_mock.call_count)
        self.assertEqual(0, west_mock.call_count)
        west_app = self.create_simple_middleware(conf=dict(region_name='west'))
        self.call(west_app, headers={'X-Auth-Token': uuid.uuid4().hex})
        self.assertEqual(1, east_mock.call_count)
        self.assertEqual(1, west_mock.call_count)