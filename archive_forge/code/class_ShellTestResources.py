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
class ShellTestResources(ShellBase):

    def setUp(self):
        super(ShellTestResources, self).setUp()
        self.set_fake_env(FAKE_ENV_KEYSTONE_V2)

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

    def test_resource_list(self):
        self._test_resource_list(True)

    def test_resource_list_no_resource_name(self):
        self._test_resource_list(False)

    def test_resource_list_empty(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'resources': []}
        stack_id = 'teststack/1'
        self.mock_request_get('/stacks/%s/resources' % stack_id, resp_dict)
        resource_list_text = self.shell('resource-list {0}'.format(stack_id))
        self.assertEqual('+---------------+----------------------+---------------+-----------------+--------------+\n| resource_name | physical_resource_id | resource_type | resource_status | updated_time |\n+---------------+----------------------+---------------+-----------------+--------------+\n+---------------+----------------------+---------------+-----------------+--------------+\n', resource_list_text)

    def _test_resource_list_more_args(self, query_args, cmd_args, response_args):
        self.register_keystone_auth_fixture()
        resp_dict = {'resources': [{'resource_name': 'foobar', 'links': [{'href': 'http://heat.example.com:8004/foo/12/resources/foobar', 'rel': 'self'}, {'href': 'http://heat.example.com:8004/foo/12', 'rel': 'stack'}]}]}
        stack_id = 'teststack/1'
        self.mock_request_get('/stacks/%s/resources?%s' % (stack_id, query_args), resp_dict)
        shell_cmd = 'resource-list %s %s' % (stack_id, cmd_args)
        resource_list_text = self.shell(shell_cmd)
        for field in response_args:
            self.assertRegex(resource_list_text, field)

    def test_resource_list_nested(self):
        self._test_resource_list_more_args(query_args='nested_depth=99', cmd_args='--nested-depth 99', response_args=['resource_name', 'foobar', 'stack_name', 'foo'])

    def test_resource_list_filter(self):
        self._test_resource_list_more_args(query_args='name=foobar', cmd_args='--filter name=foobar', response_args=['resource_name', 'foobar'])

    def test_resource_list_detail(self):
        self._test_resource_list_more_args(query_args=parse.urlencode({'with_detail': True}, True), cmd_args='--with-detail', response_args=['resource_name', 'foobar', 'stack_name', 'foo'])

    def test_resource_show_with_attrs(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'resource': {'description': '', 'links': [{'href': 'http://heat.example.com:8004/foo', 'rel': 'self'}, {'href': 'http://heat.example.com:8004/foo2', 'rel': 'resource'}], 'logical_resource_id': 'aResource', 'physical_resource_id': '43b68bae-ed5d-4aed-a99f-0b3d39c2418a', 'required_by': [], 'resource_name': 'aResource', 'resource_status': 'CREATE_COMPLETE', 'resource_status_reason': 'state changed', 'resource_type': 'OS::Nova::Server', 'updated_time': '2014-01-06T16:14:26Z', 'creation_time': '2014-01-06T16:14:26Z', 'attributes': {'attr_a': 'value_of_attr_a', 'attr_b': 'value_of_attr_b'}}}
        stack_id = 'teststack/1'
        resource_name = 'aResource'
        self.mock_request_get('/stacks/%s/resources/%s?with_attr=attr_a&with_attr=attr_b' % (parse.quote(stack_id), parse.quote(encodeutils.safe_encode(resource_name))), resp_dict)
        resource_show_text = self.shell('resource-show {0} {1} --with-attr attr_a --with-attr attr_b'.format(stack_id, resource_name))
        required = ['description', 'links', 'http://heat.example.com:8004/foo[0-9]', 'logical_resource_id', 'aResource', 'physical_resource_id', '43b68bae-ed5d-4aed-a99f-0b3d39c2418a', 'required_by', 'resource_name', 'aResource', 'resource_status', 'CREATE_COMPLETE', 'resource_status_reason', 'state changed', 'resource_type', 'OS::Nova::Server', 'updated_time', '2014-01-06T16:14:26Z']
        for r in required:
            self.assertRegex(resource_show_text, r)

    def test_resource_signal(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        resource_name = 'aResource'
        self.mock_request_post('/stacks/%s/resources/%s/signal' % (parse.quote(stack_id), parse.quote(encodeutils.safe_encode(resource_name))), '', data={'message': 'Content'})
        text = self.shell('resource-signal {0} {1} -D {{"message":"Content"}}'.format(stack_id, resource_name))
        self.assertEqual('', text)

    def test_resource_signal_no_data(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        resource_name = 'aResource'
        self.mock_request_post('/stacks/%s/resources/%s/signal' % (parse.quote(stack_id), parse.quote(encodeutils.safe_encode(resource_name))), '', data=None)
        text = self.shell('resource-signal {0} {1}'.format(stack_id, resource_name))
        self.assertEqual('', text)

    def test_resource_signal_no_json(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        resource_name = 'aResource'
        error = self.assertRaises(exc.CommandError, self.shell, 'resource-signal {0} {1} -D [2'.format(stack_id, resource_name))
        self.assertIn('Data should be in JSON format', str(error))

    def test_resource_signal_no_dict(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        resource_name = 'aResource'
        error = self.assertRaises(exc.CommandError, self.shell, 'resource-signal {0} {1} -D "message"'.format(stack_id, resource_name))
        self.assertEqual('Data should be a JSON dict', str(error))

    def test_resource_signal_both_data(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        resource_name = 'aResource'
        error = self.assertRaises(exc.CommandError, self.shell, 'resource-signal {0} {1} -D "message" -f foo'.format(stack_id, resource_name))
        self.assertEqual('Can only specify one of data and data-file', str(error))

    def test_resource_signal_data_file(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        resource_name = 'aResource'
        self.mock_request_post('/stacks/%s/resources/%s/signal' % (parse.quote(stack_id), parse.quote(encodeutils.safe_encode(resource_name))), '', data={'message': 'Content'})
        with tempfile.NamedTemporaryFile() as data_file:
            data_file.write(b'{"message":"Content"}')
            data_file.flush()
            text = self.shell('resource-signal {0} {1} -f {2}'.format(stack_id, resource_name, data_file.name))
            self.assertEqual('', text)

    def test_resource_mark_unhealthy(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        resource_name = 'aResource'
        self.mock_request_patch('/stacks/%s/resources/%s' % (parse.quote(stack_id), parse.quote(encodeutils.safe_encode(resource_name))), '', req_headers=False, data={'mark_unhealthy': True, 'resource_status_reason': 'Any'})
        text = self.shell('resource-mark-unhealthy {0} {1} Any'.format(stack_id, resource_name))
        self.assertEqual('', text)

    def test_resource_mark_unhealthy_reset(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        resource_name = 'aResource'
        self.mock_request_patch('/stacks/%s/resources/%s' % (parse.quote(stack_id), parse.quote(encodeutils.safe_encode(resource_name))), '', req_headers=False, data={'mark_unhealthy': False, 'resource_status_reason': 'Any'})
        text = self.shell('resource-mark-unhealthy --reset {0} {1} Any'.format(stack_id, resource_name))
        self.assertEqual('', text)

    def test_resource_mark_unhealthy_no_reason(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        resource_name = 'aResource'
        self.mock_request_patch('/stacks/%s/resources/%s' % (parse.quote(stack_id), parse.quote(encodeutils.safe_encode(resource_name))), '', req_headers=False, data={'mark_unhealthy': True, 'resource_status_reason': ''})
        text = self.shell('resource-mark-unhealthy {0} {1}'.format(stack_id, resource_name))
        self.assertEqual('', text)