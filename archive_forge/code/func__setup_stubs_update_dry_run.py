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
def _setup_stubs_update_dry_run(self, template_file, existing=False, show_nested=False):
    self.register_keystone_auth_fixture()
    template_data = open(template_file).read()
    replaced_res = {'resource_name': 'my_res', 'resource_identity': {'stack_name': 'teststack2', 'stack_id': '2', 'tenant': '1234', 'path': '/resources/my_res'}, 'description': '', 'stack_identity': {'stack_name': 'teststack2', 'stack_id': '2', 'tenant': '1234', 'path': ''}, 'stack_name': 'teststack2', 'creation_time': '2015-08-19T19:43:34.025507', 'resource_status': 'COMPLETE', 'updated_time': '2015-08-19T19:43:34.025507', 'resource_type': 'OS::Heat::RandomString', 'required_by': [], 'resource_status_reason': '', 'physical_resource_id': '', 'attributes': {'value': None}, 'resource_action': 'INIT', 'metadata': {}}
    resp_dict = {'resource_changes': {'deleted': [], 'unchanged': [], 'added': [], 'replaced': [replaced_res], 'updated': []}}
    expected_data = {'files': {}, 'environment': {}, 'template': jsonutils.loads(template_data), 'parameters': {'"KeyPairName': 'updated_key"'}, 'disable_rollback': False}
    if show_nested:
        path = '/stacks/teststack2/2/preview?show_nested=True'
    else:
        path = '/stacks/teststack2/2/preview'
    if existing:
        self.mock_request_patch(path, resp_dict, data=expected_data)
    else:
        self.mock_request_put(path, resp_dict, data=expected_data)