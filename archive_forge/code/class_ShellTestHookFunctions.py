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
class ShellTestHookFunctions(ShellBase):

    def setUp(self):
        super(ShellTestHookFunctions, self).setUp()
        self.set_fake_env(FAKE_ENV_KEYSTONE_V2)

    def _stub_stack_response(self, stack_id, action='CREATE', status='IN_PROGRESS'):
        resp_dict = {'stack': {'id': stack_id.split('/')[1], 'stack_name': stack_id.split('/')[0], 'stack_status': '%s_%s' % (action, status), 'creation_time': '2014-01-06T16:14:00Z'}}
        self.mock_request_get('/stacks/teststack/1', resp_dict)

    def _stub_responses(self, stack_id, nested_id, action='CREATE'):
        action_reason = 'Stack %s started' % action
        hook_reason = '%s paused until Hook pre-%s is cleared' % (action, action.lower())
        hook_clear_reason = 'Hook pre-%s is cleared' % action.lower()
        self._stub_stack_response(stack_id, action)
        ev_resp_dict = {'events': [{'id': 'p_eventid1', 'event_time': '2014-01-06T16:14:00Z', 'resource_name': None, 'resource_status_reason': action_reason}, {'id': 'p_eventid2', 'event_time': '2014-01-06T16:17:00Z', 'resource_name': 'p_res', 'resource_status_reason': hook_reason}]}
        url = '/stacks/%s/events?nested_depth=1&sort_dir=asc' % stack_id
        self.mock_request_get(url, ev_resp_dict)
        self.mock_request_get('/stacks/%s/events?sort_dir=asc' % stack_id, ev_resp_dict)
        res_resp_dict = {'resources': [{'links': [{'href': 'http://heat/foo', 'rel': 'self'}, {'href': 'http://heat/foo2', 'rel': 'resource'}, {'href': 'http://heat/%s' % nested_id, 'rel': 'nested'}], 'resource_type': 'OS::Nested::Foo'}]}
        self.mock_request_get('/stacks/%s/resources' % stack_id, res_resp_dict)
        nev_resp_dict = {'events': [{'id': 'n_eventid1', 'event_time': '2014-01-06T16:15:00Z', 'resource_name': 'n_res', 'resource_status_reason': hook_reason}, {'id': 'n_eventid2', 'event_time': '2014-01-06T16:16:00Z', 'resource_name': 'n_res', 'resource_status_reason': hook_clear_reason}]}
        self.mock_request_get('/stacks/%s/events?sort_dir=asc' % nested_id, nev_resp_dict)

    def test_hook_poll_pre_create(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        nested_id = 'nested/2'
        self._stub_responses(stack_id, nested_id, 'CREATE')
        list_text = self.shell('hook-poll %s --nested-depth 1' % stack_id)
        hook_reason = 'CREATE paused until Hook pre-create is cleared'
        required = ['id', 'p_eventid2', 'stack_name', 'teststack', hook_reason]
        for r in required:
            self.assertRegex(list_text, r)
        self.assertNotRegex(list_text, 'p_eventid1')
        self.assertNotRegex(list_text, 'n_eventid1')
        self.assertNotRegex(list_text, 'n_eventid2')

    def test_hook_poll_pre_update(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        nested_id = 'nested/2'
        self._stub_responses(stack_id, nested_id, 'UPDATE')
        list_text = self.shell('hook-poll %s --nested-depth 1' % stack_id)
        hook_reason = 'UPDATE paused until Hook pre-update is cleared'
        required = ['id', 'p_eventid2', 'stack_name', 'teststack', hook_reason]
        for r in required:
            self.assertRegex(list_text, r)
        self.assertNotRegex(list_text, 'p_eventid1')
        self.assertNotRegex(list_text, 'n_eventid1')
        self.assertNotRegex(list_text, 'n_eventid2')

    def test_hook_poll_pre_delete(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        nested_id = 'nested/2'
        self._stub_responses(stack_id, nested_id, 'DELETE')
        list_text = self.shell('hook-poll %s --nested-depth 1' % stack_id)
        hook_reason = 'DELETE paused until Hook pre-delete is cleared'
        required = ['id', 'p_eventid2', 'stack_name', 'teststack', hook_reason]
        for r in required:
            self.assertRegex(list_text, r)
        self.assertNotRegex(list_text, 'p_eventid1')
        self.assertNotRegex(list_text, 'n_eventid1')
        self.assertNotRegex(list_text, 'n_eventid2')

    def test_hook_poll_bad_status(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        self._stub_stack_response(stack_id, status='COMPLETE')
        error = self.assertRaises(exc.CommandError, self.shell, 'hook-poll %s --nested-depth 1' % stack_id)
        self.assertIn('Stack status CREATE_COMPLETE not IN_PROGRESS', str(error))

    def test_shell_nested_depth_invalid_value(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        error = self.assertRaises(exc.CommandError, self.shell, 'hook-poll %s --nested-depth Z' % stack_id)
        self.assertIn('--nested-depth invalid value Z', str(error))

    def test_hook_poll_clear_bad_status(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        self._stub_stack_response(stack_id, status='COMPLETE')
        error = self.assertRaises(exc.CommandError, self.shell, 'hook-clear %s aresource' % stack_id)
        self.assertIn('Stack status CREATE_COMPLETE not IN_PROGRESS', str(error))

    def test_hook_poll_clear_bad_action(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        self._stub_stack_response(stack_id, action='BADACTION')
        error = self.assertRaises(exc.CommandError, self.shell, 'hook-clear %s aresource' % stack_id)
        self.assertIn('Unexpected stack status BADACTION_IN_PROGRESS', str(error))