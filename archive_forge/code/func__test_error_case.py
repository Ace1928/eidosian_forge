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
def _test_error_case(self, code, message):
    self.register_keystone_auth_fixture()
    resp_dict = {'explanation': '', 'code': code, 'error': {'message': message, 'type': '', 'traceback': ''}, 'title': 'test title'}
    resp_string = jsonutils.dumps(resp_dict)
    resp = fakes.FakeHTTPResponse(code, 'test reason', {'content-type': 'application/json'}, resp_string)
    self.mock_request_error('/services', 'GET', exc.from_response(resp))
    exc.verbose = 1
    e = self.assertRaises(exc.HTTPException, self.shell, 'service-list')
    self.assertIn(message, str(e))