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
def _output_fake_response(self, output_key):
    outputs = [{'output_value': 'value1', 'output_key': 'output1', 'description': 'test output 1'}, {'output_value': ['output', 'value', '2'], 'output_key': 'output2', 'description': 'test output 2'}, {'output_value': u'test♥', 'output_key': 'output_uni', 'description': 'test output unicode'}]

    def find_output(key):
        for out in outputs:
            if out['output_key'] == key:
                return {'output': out}
    self.mock_request_get('/stacks/teststack/1/outputs/%s' % output_key, find_output(output_key))