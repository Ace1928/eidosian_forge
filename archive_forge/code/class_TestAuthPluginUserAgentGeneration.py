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
class TestAuthPluginUserAgentGeneration(BaseAuthTokenMiddlewareTest):

    def setUp(self):
        super(TestAuthPluginUserAgentGeneration, self).setUp()
        self.auth_url = uuid.uuid4().hex
        self.project_id = uuid.uuid4().hex
        self.username = uuid.uuid4().hex
        self.password = uuid.uuid4().hex
        self.section = uuid.uuid4().hex
        self.user_domain_id = uuid.uuid4().hex
        loading.register_auth_conf_options(self.cfg.conf, group=self.section)
        opts = loading.get_auth_plugin_conf_options('password')
        self.cfg.register_opts(opts, group=self.section)
        loading.register_auth_conf_options(self.cfg.conf, group=_base.AUTHTOKEN_GROUP)
        self.cfg.config(auth_section=self.section, group=_base.AUTHTOKEN_GROUP)
        self.cfg.config(auth_type='password', password=self.password, project_id=self.project_id, user_domain_id=self.user_domain_id, group=self.section)

    def test_no_project_configured(self):
        ksm_version = uuid.uuid4().hex
        conf = {'username': self.username, 'auth_url': self.auth_url}
        app = self._create_app(conf, '', ksm_version)
        self._assert_user_agent(app, '', ksm_version)

    def test_project_in_configuration(self):
        project = uuid.uuid4().hex
        project_version = uuid.uuid4().hex
        ksm_version = uuid.uuid4().hex
        conf = {'username': self.username, 'auth_url': self.auth_url, 'project': project}
        app = self._create_app(conf, project_version, ksm_version)
        project_with_version = '{0}/{1} '.format(project, project_version)
        self._assert_user_agent(app, project_with_version, ksm_version)

    def test_project_not_installed_results_in_unknown_version(self):
        project = uuid.uuid4().hex
        conf = {'username': self.username, 'auth_url': self.auth_url, 'project': project}
        v = pbr.version.VersionInfo('keystonemiddleware').version_string()
        app = self.create_simple_middleware(conf=conf, use_global_conf=True)
        project_with_version = '{0}/{1} '.format(project, 'unknown')
        self._assert_user_agent(app, project_with_version, v)

    def test_project_in_oslo_configuration(self):
        project = uuid.uuid4().hex
        project_version = uuid.uuid4().hex
        ksm_version = uuid.uuid4().hex
        conf = {'username': self.username, 'auth_url': self.auth_url}
        with mock.patch.object(self.cfg.conf, 'project', new=project, create=True):
            app = self._create_app(conf, project_version, ksm_version)
        project = '{0}/{1} '.format(project, project_version)
        self._assert_user_agent(app, project, ksm_version)

    def _create_app(self, conf, project_version, ksm_version):
        fake_pkg_resources = mock.Mock()
        fake_pkg_resources.get_distribution().version = project_version
        fake_version_info = mock.Mock()
        fake_version_info.version_string.return_value = ksm_version
        fake_pbr_version = mock.Mock()
        fake_pbr_version.VersionInfo.return_value = fake_version_info
        body = uuid.uuid4().hex
        at_pbr = 'keystonemiddleware._common.config.pbr.version'
        with mock.patch('keystonemiddleware._common.config.pkg_resources', new=fake_pkg_resources):
            with mock.patch(at_pbr, new=fake_pbr_version):
                return self.create_simple_middleware(body=body, conf=conf)

    def _assert_user_agent(self, app, project, ksm_version):
        sess = app._identity_server._adapter.session
        expected_ua = '{0}keystonemiddleware.auth_token/{1}'.format(project, ksm_version)
        self.assertThat(sess.user_agent, matchers.StartsWith(expected_ua))