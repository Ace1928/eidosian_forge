import importlib
import os
from unittest import mock
from osc_lib.tests import utils as osc_lib_test_utils
import wrapt
from openstackclient import shell
class TestShellTokenEndpointAuthEnv(TestShell):

    def setUp(self):
        super(TestShellTokenEndpointAuthEnv, self).setUp()
        env = {'OS_TOKEN': DEFAULT_TOKEN, 'OS_ENDPOINT': DEFAULT_SERVICE_URL}
        self.useFixture(osc_lib_test_utils.EnvFixture(env.copy()))

    def test_env(self):
        flag = ''
        kwargs = {'token': DEFAULT_TOKEN, 'endpoint': DEFAULT_SERVICE_URL}
        self._assert_admin_token_auth(flag, kwargs)

    def test_only_token(self):
        flag = '--os-token xyzpdq'
        kwargs = {'token': 'xyzpdq', 'endpoint': DEFAULT_SERVICE_URL}
        self._assert_token_auth(flag, kwargs)

    def test_only_url(self):
        flag = '--os-endpoint http://cloud.local:555'
        kwargs = {'token': DEFAULT_TOKEN, 'endpoint': 'http://cloud.local:555'}
        self._assert_token_auth(flag, kwargs)

    def test_empty_auth(self):
        os.environ = {}
        flag = ''
        kwargs = {'token': '', 'endpoint': ''}
        self._assert_token_auth(flag, kwargs)