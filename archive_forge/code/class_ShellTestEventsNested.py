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
class ShellTestEventsNested(ShellBase):

    def setUp(self):
        super(ShellTestEventsNested, self).setUp()
        self.set_fake_env(FAKE_ENV_KEYSTONE_V2)

    def test_shell_nested_depth_invalid_xor(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        resource_name = 'aResource'
        error = self.assertRaises(exc.CommandError, self.shell, 'event-list {0} --resource {1} --nested-depth 5'.format(stack_id, resource_name))
        self.assertIn('--nested-depth cannot be specified with --resource', str(error))

    def test_shell_nested_depth_invalid_value(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        error = self.assertRaises(exc.CommandError, self.shell, 'event-list {0} --nested-depth Z'.format(stack_id))
        self.assertIn('--nested-depth invalid value Z', str(error))

    def test_shell_nested_depth_zero(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'events': [{'id': 'eventid1'}, {'id': 'eventid2'}]}
        stack_id = 'teststack/1'
        self.mock_request_get('/stacks/%s/events?sort_dir=asc' % stack_id, resp_dict)
        list_text = self.shell('event-list %s --nested-depth 0' % stack_id)
        required = ['id', 'eventid1', 'eventid2']
        for r in required:
            self.assertRegex(list_text, r)

    def _stub_event_list_response_old_api(self, stack_id, nested_id, timestamps, first_request):
        ev_resp_dict = {'events': [{'id': 'p_eventid1', 'event_time': timestamps[0]}, {'id': 'p_eventid2', 'event_time': timestamps[3]}]}
        self.mock_request_get(first_request, ev_resp_dict)
        self.mock_request_get('/stacks/%s/events?sort_dir=asc' % stack_id, ev_resp_dict)
        res_resp_dict = {'resources': [{'links': [{'href': 'http://heat/foo', 'rel': 'self'}, {'href': 'http://heat/foo2', 'rel': 'resource'}, {'href': 'http://heat/%s' % nested_id, 'rel': 'nested'}], 'resource_type': 'OS::Nested::Foo'}]}
        self.mock_request_get('/stacks/%s/resources' % stack_id, res_resp_dict)
        nev_resp_dict = {'events': [{'id': 'n_eventid1', 'event_time': timestamps[1]}, {'id': 'n_eventid2', 'event_time': timestamps[2]}]}
        self.mock_request_get('/stacks/%s/events?sort_dir=asc' % nested_id, nev_resp_dict)

    def test_shell_nested_depth_old_api(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        nested_id = 'nested/2'
        timestamps = ('2014-01-06T16:14:00Z', '2014-01-06T16:15:00Z', '2014-01-06T16:16:00Z', '2014-01-06T16:17:00Z')
        first_request = '/stacks/%s/events?nested_depth=1&sort_dir=asc' % stack_id
        self._stub_event_list_response_old_api(stack_id, nested_id, timestamps, first_request)
        list_text = self.shell('event-list %s --nested-depth 1' % stack_id)
        required = ['id', 'p_eventid1', 'p_eventid2', 'n_eventid1', 'n_eventid2', 'stack_name', 'teststack', 'nested']
        for r in required:
            self.assertRegex(list_text, r)
        self.assertRegex(list_text, '%s.*\n.*%s.*\n.*%s.*\n.*%s' % timestamps)

    def test_shell_nested_depth_marker_old_api(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        nested_id = 'nested/2'
        timestamps = ('2014-01-06T16:14:00Z', '2014-01-06T16:15:00Z', '2014-01-06T16:16:00Z', '2014-01-06T16:17:00Z')
        first_request = '/stacks/%s/events?marker=n_eventid1&nested_depth=1&sort_dir=asc' % stack_id
        self._stub_event_list_response_old_api(stack_id, nested_id, timestamps, first_request)
        list_text = self.shell('event-list %s --nested-depth 1 --marker n_eventid1' % stack_id)
        required = ['id', 'p_eventid2', 'n_eventid1', 'n_eventid2', 'stack_name', 'teststack', 'nested']
        for r in required:
            self.assertRegex(list_text, r)
        self.assertNotRegex(list_text, 'p_eventid1')
        self.assertRegex(list_text, '%s.*\n.*%s.*\n.*%s.*' % timestamps[1:])

    def test_shell_nested_depth_limit_old_api(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        nested_id = 'nested/2'
        timestamps = ('2014-01-06T16:14:00Z', '2014-01-06T16:15:00Z', '2014-01-06T16:16:00Z', '2014-01-06T16:17:00Z')
        first_request = '/stacks/%s/events?limit=2&nested_depth=1&sort_dir=asc' % stack_id
        self._stub_event_list_response_old_api(stack_id, nested_id, timestamps, first_request)
        list_text = self.shell('event-list %s --nested-depth 1 --limit 2' % stack_id)
        required = ['id', 'p_eventid1', 'n_eventid1', 'stack_name', 'teststack', 'nested']
        for r in required:
            self.assertRegex(list_text, r)
        self.assertNotRegex(list_text, 'p_eventid2')
        self.assertNotRegex(list_text, 'n_eventid2')
        self.assertRegex(list_text, '%s.*\n.*%s.*\n' % timestamps[:2])

    def _nested_events(self):
        links = [{'rel': 'self'}, {'rel': 'resource'}, {'rel': 'stack'}, {'rel': 'root_stack'}]
        return [{'id': 'p_eventid1', 'event_time': '2014-01-06T16:14:00Z', 'stack_id': '1', 'resource_name': 'the_stack', 'resource_status': 'CREATE_IN_PROGRESS', 'resource_status_reason': 'Stack CREATE started', 'links': links}, {'id': 'n_eventid1', 'event_time': '2014-01-06T16:15:00Z', 'stack_id': '2', 'resource_name': 'nested_stack', 'resource_status': 'CREATE_IN_PROGRESS', 'resource_status_reason': 'Stack CREATE started', 'links': links}, {'id': 'n_eventid2', 'event_time': '2014-01-06T16:16:00Z', 'stack_id': '2', 'resource_name': 'nested_stack', 'resource_status': 'CREATE_COMPLETE', 'resource_status_reason': 'Stack CREATE completed', 'links': links}, {'id': 'p_eventid2', 'event_time': '2014-01-06T16:17:00Z', 'stack_id': '1', 'resource_name': 'the_stack', 'resource_status': 'CREATE_COMPLETE', 'resource_status_reason': 'Stack CREATE completed', 'links': links}]

    def test_shell_nested_depth(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        nested_events = self._nested_events()
        ev_resp_dict = {'events': nested_events}
        url = '/stacks/%s/events?nested_depth=1&sort_dir=asc' % stack_id
        self.mock_request_get(url, ev_resp_dict)
        list_text = self.shell('event-list %s --nested-depth 1 --format log' % stack_id)
        self.assertEqual('2014-01-06 16:14:00Z [the_stack]: CREATE_IN_PROGRESS  Stack CREATE started\n2014-01-06 16:15:00Z [nested_stack]: CREATE_IN_PROGRESS  Stack CREATE started\n2014-01-06 16:16:00Z [nested_stack]: CREATE_COMPLETE  Stack CREATE completed\n2014-01-06 16:17:00Z [the_stack]: CREATE_COMPLETE  Stack CREATE completed\n', list_text)

    def test_shell_nested_depth_marker(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        nested_events = self._nested_events()
        ev_resp_dict = {'events': nested_events[1:]}
        url = '/stacks/%s/events?marker=n_eventid1&nested_depth=1&sort_dir=asc' % stack_id
        self.mock_request_get(url, ev_resp_dict)
        list_text = self.shell('event-list %s --nested-depth 1 --format log --marker n_eventid1' % stack_id)
        self.assertEqual('2014-01-06 16:15:00Z [nested_stack]: CREATE_IN_PROGRESS  Stack CREATE started\n2014-01-06 16:16:00Z [nested_stack]: CREATE_COMPLETE  Stack CREATE completed\n2014-01-06 16:17:00Z [the_stack]: CREATE_COMPLETE  Stack CREATE completed\n', list_text)

    def test_shell_nested_depth_limit(self):
        self.register_keystone_auth_fixture()
        stack_id = 'teststack/1'
        nested_events = self._nested_events()
        ev_resp_dict = {'events': nested_events[:2]}
        url = '/stacks/%s/events?limit=2&nested_depth=1&sort_dir=asc' % stack_id
        self.mock_request_get(url, ev_resp_dict)
        list_text = self.shell('event-list %s --nested-depth 1 --format log --limit 2' % stack_id)
        self.assertEqual('2014-01-06 16:14:00Z [the_stack]: CREATE_IN_PROGRESS  Stack CREATE started\n2014-01-06 16:15:00Z [nested_stack]: CREATE_IN_PROGRESS  Stack CREATE started\n', list_text)