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
class ShellTestManageService(ShellBase):

    def setUp(self):
        super(ShellTestManageService, self).setUp()
        self.set_fake_env(FAKE_ENV_KEYSTONE_V2)

    def _set_fake_env(self):
        """Patch os.environ to avoid required auth info."""
        self.set_fake_env(FAKE_ENV_KEYSTONE_V2)

    def _test_error_case(self, code, message):
        self.register_keystone_auth_fixture()
        resp_dict = {'explanation': '', 'code': code, 'error': {'message': message, 'type': '', 'traceback': ''}, 'title': 'test title'}
        resp_string = jsonutils.dumps(resp_dict)
        resp = fakes.FakeHTTPResponse(code, 'test reason', {'content-type': 'application/json'}, resp_string)
        self.mock_request_error('/services', 'GET', exc.from_response(resp))
        exc.verbose = 1
        e = self.assertRaises(exc.HTTPException, self.shell, 'service-list')
        self.assertIn(message, str(e))

    def test_service_list(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'services': [{'status': 'up', 'binary': 'heat-engine', 'engine_id': '9d9242c3-4b9e-45e1-9e74-7615fbf20e5d', 'hostname': 'mrkanag', 'updated_at': '2015-02-03T05:57:59.000000', 'topic': 'engine', 'host': 'engine-1'}]}
        self.mock_request_get('/services', resp_dict)
        services_text = self.shell('service-list')
        required = ['hostname', 'binary', 'engine_id', 'host', 'topic', 'updated_at', 'status']
        for r in required:
            self.assertRegex(services_text, r)

    def test_service_list_503(self):
        self._test_error_case(message='All heat engines are down', code=503)

    def test_service_list_403(self):
        self._test_error_case(message='You are not authorized to complete this action', code=403)