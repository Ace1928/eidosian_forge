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
def _test_resource_list(self, with_resource_name):
    self.register_keystone_auth_fixture()
    resp_dict = {'resources': [{'links': [{'href': 'http://heat.example.com:8004/foo', 'rel': 'self'}, {'href': 'http://heat.example.com:8004/foo2', 'rel': 'resource'}], 'logical_resource_id': 'aLogicalResource', 'physical_resource_id': '43b68bae-ed5d-4aed-a99f-0b3d39c2418a', 'resource_status': 'CREATE_COMPLETE', 'resource_status_reason': 'state changed', 'resource_type': 'OS::Nova::Server', 'updated_time': '2014-01-06T16:14:26Z'}]}
    if with_resource_name:
        resp_dict['resources'][0]['resource_name'] = 'aResource'
    stack_id = 'teststack/1'
    self.mock_request_get('/stacks/%s/resources' % stack_id, resp_dict)
    resource_list_text = self.shell('resource-list {0}'.format(stack_id))
    required = ['physical_resource_id', 'resource_type', 'resource_status', 'updated_time', '43b68bae-ed5d-4aed-a99f-0b3d39c2418a', 'OS::Nova::Server', 'CREATE_COMPLETE', '2014-01-06T16:14:26Z']
    if with_resource_name:
        required.append('resource_name')
        required.append('aResource')
    else:
        required.append('logical_resource_id')
        required.append('aLogicalResource')
    for r in required:
        self.assertRegex(resource_list_text, r)