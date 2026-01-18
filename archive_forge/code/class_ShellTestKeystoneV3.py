import io
import re
import sys
from unittest import mock
import fixtures
from keystoneauth1 import fixture
from testtools import matchers
from zunclient import api_versions
from zunclient import exceptions
import zunclient.shell
from zunclient.tests.unit import utils
class ShellTestKeystoneV3(ShellTest):
    AUTH_URL = 'http://no.where/v3'

    def make_env(self, exclude=None, fake_env=FAKE_ENV):
        if 'OS_AUTH_URL' in fake_env:
            fake_env.update({'OS_AUTH_URL': self.AUTH_URL})
        env = dict(((k, v) for k, v in fake_env.items() if k != exclude))
        self.useFixture(fixtures.MonkeyPatch('os.environ', env))

    def register_keystone_discovery_fixture(self, mreq):
        v3_url = 'http://no.where/v3'
        v3_version = fixture.V3Discovery(v3_url)
        mreq.register_uri('GET', v3_url, json=_create_ver_list([v3_version]), status_code=200)

    @mock.patch('zunclient.client.Client')
    def test_main_endpoint_public(self, mock_client):
        self.make_env(fake_env=FAKE_ENV4)
        self.shell('--zun-api-version 1.29 --endpoint-type publicURL service-list')
        mock_client.assert_called_once_with(username='username', password='password', interface='publicURL', project_id='project_id', project_name=None, auth_url=self.AUTH_URL, service_type='container', region_name=None, project_domain_id='', project_domain_name='Default', user_domain_id='', user_domain_name='Default', endpoint_override=None, insecure=False, profile=None, cacert=None, cert=None, key=None, version=api_versions.APIVersion('1.29'))