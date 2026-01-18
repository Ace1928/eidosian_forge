import os
import sys
import tempfile
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import fixture as keystone_fixture
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from requests_mock.contrib import fixture as rm_fixture
import testscenarios
import testtools
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
import heatclient.shell
from heatclient.tests.unit import fakes
import heatclient.v1.shell
class ShellParamValidationTest(TestCase):
    scenarios = [('stack-create', dict(command='stack-create ts -P "ab"', with_tmpl=True, err='Malformed parameter')), ('stack-update', dict(command='stack-update ts -P "a-b"', with_tmpl=True, err='Malformed parameter')), ('stack-list-with-sort-dir', dict(command='stack-list --sort-dir up', with_tmpl=False, err='Sorting direction must be one of')), ('stack-list-with-sort-key', dict(command='stack-list --sort-keys owner', with_tmpl=False, err="Sorting key 'owner' not one of"))]

    def test_bad_parameters(self):
        self.register_keystone_auth_fixture()
        fake_env = {'OS_USERNAME': 'username', 'OS_PASSWORD': 'password', 'OS_TENANT_NAME': 'tenant_name', 'OS_AUTH_URL': BASE_URL}
        self.set_fake_env(fake_env)
        cmd = self.command
        if self.with_tmpl:
            template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
            cmd = '%s --template-file=%s ' % (self.command, template_file)
        self.shell_error(cmd, self.err, exception=exc.CommandError)