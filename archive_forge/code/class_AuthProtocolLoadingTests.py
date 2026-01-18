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
class AuthProtocolLoadingTests(BaseAuthTokenMiddlewareTest):
    AUTH_URL = 'http://auth.url/prefix'
    DISC_URL = 'http://disc.url/prefix'
    KEYSTONE_BASE_URL = 'http://keystone.url/prefix'
    CRUD_URL = 'http://crud.url/prefix'
    KEYSTONE_URL = KEYSTONE_BASE_URL + '/v3'

    def setUp(self):
        super(AuthProtocolLoadingTests, self).setUp()
        self.project_id = uuid.uuid4().hex
        self.requests_mock.get(self.AUTH_URL, json=fixture.DiscoveryList(href=self.DISC_URL), status_code=300)
        self.requests_mock.get(self.KEYSTONE_BASE_URL + '/', json=fixture.DiscoveryList(href=self.CRUD_URL), status_code=300)

    def good_request(self, app):
        admin_token_id = uuid.uuid4().hex
        admin_token = fixture.V3Token(project_id=self.project_id)
        s = admin_token.add_service('identity', name='keystone')
        s.add_standard_endpoints(internal=self.KEYSTONE_URL)
        self.requests_mock.post('%s/v3/auth/tokens' % self.AUTH_URL, json=admin_token, headers={'X-Subject-Token': admin_token_id})
        user_token_id = uuid.uuid4().hex
        user_token = fixture.V3Token()
        user_token.set_project_scope()
        request_headers = {'X-Subject-Token': user_token_id, 'X-Auth-Token': admin_token_id}
        self.requests_mock.get('%s/v3/auth/tokens' % self.KEYSTONE_BASE_URL, request_headers=request_headers, json=user_token, headers={'X-Subject-Token': uuid.uuid4().hex})
        resp = self.call(app, headers={'X-Auth-Token': user_token_id})
        return resp

    def test_loading_password_plugin(self):
        opts = loading.get_auth_plugin_conf_options('password')
        self.cfg.register_opts(opts, group=_base.AUTHTOKEN_GROUP)
        project_id = uuid.uuid4().hex
        loading.register_auth_conf_options(self.cfg.conf, group=_base.AUTHTOKEN_GROUP)
        self.cfg.config(auth_type='password', username='testuser', password='testpass', auth_url=self.AUTH_URL, project_id=project_id, user_domain_id='userdomainid', group=_base.AUTHTOKEN_GROUP)
        body = uuid.uuid4().hex
        app = self.create_simple_middleware(body=body)
        resp = self.good_request(app)
        self.assertEqual(body.encode(), resp.body)

    @staticmethod
    def get_plugin(app):
        return app._identity_server._adapter.auth

    def test_invalid_plugin_fails_to_initialize(self):
        loading.register_auth_conf_options(self.cfg.conf, group=_base.AUTHTOKEN_GROUP)
        self.cfg.config(auth_type=uuid.uuid4().hex, group=_base.AUTHTOKEN_GROUP)
        self.assertRaises(ksa_exceptions.NoMatchingPlugin, self.create_simple_middleware)

    def test_plugin_loading_mixed_opts(self):
        opts = loading.get_auth_plugin_conf_options('password')
        self.cfg.register_opts(opts, group=_base.AUTHTOKEN_GROUP)
        username = 'testuser'
        password = 'testpass'
        loading.register_auth_conf_options(self.cfg.conf, group=_base.AUTHTOKEN_GROUP)
        self.cfg.config(auth_type='password', auth_url='http://keystone.test:5000', password=password, project_id=self.project_id, user_domain_id='userdomainid', group=_base.AUTHTOKEN_GROUP)
        conf = {'username': username, 'auth_url': self.AUTH_URL}
        body = uuid.uuid4().hex
        app = self.create_simple_middleware(body=body, conf=conf)
        resp = self.good_request(app)
        self.assertEqual(body.encode(), resp.body)
        plugin = self.get_plugin(app)
        self.assertEqual(self.AUTH_URL, plugin.auth_url)
        self.assertEqual(username, plugin._username)
        self.assertEqual(password, plugin._password)
        self.assertEqual(self.project_id, plugin._project_id)

    def test_plugin_loading_with_auth_section(self):
        section = 'testsection'
        username = 'testuser'
        password = 'testpass'
        loading.register_auth_conf_options(self.cfg.conf, group=section)
        opts = loading.get_auth_plugin_conf_options('password')
        self.cfg.register_opts(opts, group=section)
        loading.register_auth_conf_options(self.cfg.conf, group=_base.AUTHTOKEN_GROUP)
        self.cfg.config(auth_section=section, group=_base.AUTHTOKEN_GROUP)
        self.cfg.config(auth_type='password', auth_url=self.AUTH_URL, password=password, project_id=self.project_id, user_domain_id='userdomainid', group=section)
        conf = {'username': username}
        body = uuid.uuid4().hex
        app = self.create_simple_middleware(body=body, conf=conf)
        resp = self.good_request(app)
        self.assertEqual(body.encode(), resp.body)
        plugin = self.get_plugin(app)
        self.assertEqual(self.AUTH_URL, plugin.auth_url)
        self.assertEqual(username, plugin._username)
        self.assertEqual(password, plugin._password)
        self.assertEqual(self.project_id, plugin._project_id)