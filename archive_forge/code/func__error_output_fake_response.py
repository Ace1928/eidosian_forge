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
def _error_output_fake_response(self, output_key):
    resp_dict = {'output': {'output_value': 'null', 'output_key': 'output1', 'description': 'test output 1', 'output_error': 'The Referenced Attribute (0 PublicIP) is incorrect.'}}
    self.mock_request_get('/stacks/teststack/1/outputs/%s' % output_key, resp_dict)