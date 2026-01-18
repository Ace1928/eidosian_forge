import copy
import io
from unittest import mock
from osc_lib import exceptions as exc
from osc_lib import utils
import testscenarios
import yaml
from heatclient.common import template_format
from heatclient import exc as heat_exc
from heatclient.osc.v1 import stack
from heatclient.tests import inline_templates
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import events
from heatclient.v1 import resources
from heatclient.v1 import stacks
class TestStackHookClear(TestStack):
    stack = stacks.Stack(None, {'id': '1234', 'stack_name': 'my_stack', 'creation_time': '2013-08-04T20:57:55Z', 'updated_time': '2013-08-04T20:57:55Z', 'stack_status': 'CREATE_IN_PROGRESS'})
    resource = resources.Resource(None, {'stack_id': 'my_stack', 'resource_name': 'resource'})

    def setUp(self):
        super(TestStackHookClear, self).setUp()
        self.cmd = stack.StackHookClear(self.app, None)
        self.mock_client.stacks.get.return_value = self.stack
        self.mock_client.resources.list.return_value = [self.resource]

    def test_hook_clear(self):
        arglist = ['my_stack', 'resource']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.mock_client.resources.signal.assert_called_once_with(data={'unset_hook': 'pre-create'}, resource_name='resource', stack_id='my_stack')

    def test_hook_clear_pre_create(self):
        arglist = ['my_stack', 'resource', '--pre-create']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.mock_client.resources.signal.assert_called_once_with(data={'unset_hook': 'pre-create'}, resource_name='resource', stack_id='my_stack')

    def test_hook_clear_pre_update(self):
        arglist = ['my_stack', 'resource', '--pre-update']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.mock_client.resources.signal.assert_called_once_with(data={'unset_hook': 'pre-update'}, resource_name='resource', stack_id='my_stack')

    def test_hook_clear_pre_delete(self):
        arglist = ['my_stack', 'resource', '--pre-delete']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.mock_client.resources.signal.assert_called_once_with(data={'unset_hook': 'pre-delete'}, resource_name='resource', stack_id='my_stack')