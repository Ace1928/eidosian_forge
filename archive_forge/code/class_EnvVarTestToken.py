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
class EnvVarTestToken(TestCase):
    scenarios = [('tenant_id', dict(remove='OS_TENANT_ID', err='You must provide a tenant id')), ('auth_url', dict(remove='OS_AUTH_URL', err='You must provide an auth url'))]

    def test_missing_auth(self):
        fake_env = {'OS_AUTH_TOKEN': 'atoken', 'OS_TENANT_ID': 'tenant_id', 'OS_AUTH_URL': 'http://no.where'}
        fake_env[self.remove] = None
        self.set_fake_env(fake_env)
        self.shell_error('stack-list', self.err, exception=exc.CommandError)