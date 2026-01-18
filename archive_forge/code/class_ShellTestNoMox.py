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
class ShellTestNoMox(ShellTestNoMoxBase):

    def test_stack_create_parameter_missing_err_msg(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'error': {'message': 'The Parameter (key_name) was not provided.', 'type': 'UserParameterMissing'}}
        self.requests.post('http://heat.example.com/stacks', status_code=400, headers={'Content-Type': 'application/json'}, json=resp_dict)
        template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
        self.shell_error('stack-create -f %s stack' % template_file, 'The Parameter \\(key_name\\) was not provided.', exception=exc.HTTPBadRequest)

    def test_event_list(self):
        eventid1 = uuid.uuid4().hex
        eventid2 = uuid.uuid4().hex
        self.register_keystone_auth_fixture()
        h = {'Content-Type': 'text/plain; charset=UTF-8', 'location': 'http://heat.example.com/stacks/myStack/60f83b5e'}
        self.requests.get('http://heat.example.com/stacks/myStack', status_code=302, headers=h)
        resp_dict = self.event_list_resp_dict(resource_name='myDeployment', rsrc_eventid1=eventid1, rsrc_eventid2=eventid2)
        self.requests.get('http://heat.example.com/stacks/myStack/60f83b5e/resources/myDeployment/events', headers={'Content-Type': 'application/json'}, json=resp_dict)
        list_text = self.shell('event-list -r myDeployment myStack')
        required = ['resource_name', 'id', 'resource_status_reason', 'resource_status', 'event_time', 'myDeployment', eventid1, eventid2, 'state changed', 'CREATE_IN_PROGRESS', '2013-12-05T14:14:31', '2013-12-05T14:14:32']
        for r in required:
            self.assertRegex(list_text, r)