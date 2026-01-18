import contextlib
import copy
import json as jsonutils
import os
from unittest import mock
from cliff import columns as cliff_columns
import fixtures
from keystoneauth1 import loading
from openstack.config import cloud_region
from openstack.config import defaults
from oslo_utils import importutils
from requests_mock.contrib import fixture
import testtools
from osc_lib import clientmanager
from osc_lib import shell
from osc_lib.tests import fakes
class TestClientManager(TestCase):
    """ClientManager class test framework"""
    default_password_auth = {'auth_url': fakes.AUTH_URL, 'username': fakes.USERNAME, 'password': fakes.PASSWORD, 'project_name': fakes.PROJECT_NAME}
    default_token_auth = {'auth_url': fakes.AUTH_URL, 'token': fakes.AUTH_TOKEN}

    def setUp(self):
        super(TestClientManager, self).setUp()
        self.mock = mock.Mock()
        self.requests = self.useFixture(fixture.Fixture())
        self.stub_auth(json=fakes.TEST_RESPONSE_DICT)
        self.stub_auth(json=fakes.TEST_RESPONSE_DICT, url='/'.join([fakes.AUTH_URL, 'v2.0/tokens']))
        self.stub_auth(json=fakes.TEST_RESPONSE_DICT_V3, url='/'.join([fakes.AUTH_URL, 'v3/auth/tokens']))
        self.stub_auth(json=fakes.TEST_RESPONSE_DICT_V3, url='/'.join([fakes.AUTH_URL, 'auth/tokens']))
        self.stub_auth(json=fakes.TEST_VERSIONS, url=fakes.AUTH_URL, verb='GET')
        self.auth_mock = mock.Mock()

    def stub_auth(self, json=None, url=None, verb=None, **kwargs):
        subject_token = fakes.AUTH_TOKEN
        base_url = fakes.AUTH_URL
        if json:
            text = jsonutils.dumps(json)
            headers = {'X-Subject-Token': subject_token, 'Content-Type': 'application/json'}
        if not url:
            url = '/'.join([base_url, 'tokens'])
        url = url.replace('/?', '?')
        if not verb:
            verb = 'POST'
        self.requests.register_uri(verb, url, headers=headers, text=text)

    def _clientmanager_class(self):
        """Allow subclasses to override the ClientManager class"""
        return clientmanager.ClientManager

    def _make_clientmanager(self, auth_args=None, config_args=None, identity_api_version=None, auth_plugin_name=None, auth_required=None):
        if identity_api_version is None:
            identity_api_version = '2.0'
        if auth_plugin_name is None:
            auth_plugin_name = 'password'
        if auth_plugin_name.endswith('password'):
            auth_dict = copy.deepcopy(self.default_password_auth)
        elif auth_plugin_name.endswith('token'):
            auth_dict = copy.deepcopy(self.default_token_auth)
        else:
            auth_dict = {}
        if auth_args is not None:
            auth_dict = auth_args
        cli_options = defaults.get_defaults()
        cli_options.update({'auth_type': auth_plugin_name, 'auth': auth_dict, 'interface': fakes.INTERFACE, 'region_name': fakes.REGION_NAME})
        if config_args is not None:
            cli_options.update(config_args)
        loader = loading.get_plugin_loader(auth_plugin_name)
        auth_plugin = loader.load_from_options(**auth_dict)
        client_manager = self._clientmanager_class()(cli_options=cloud_region.CloudRegion(name='t1', region_name='1', config=cli_options, auth_plugin=auth_plugin), api_version={'identity': identity_api_version})
        client_manager._auth_required = auth_required is True
        client_manager.setup_auth()
        client_manager.auth_ref
        self.assertEqual(auth_plugin_name, client_manager.auth_plugin_name)
        return client_manager