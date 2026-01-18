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
class _TestStackCheckBase(object):
    stack = stacks.Stack(None, {'id': '1234', 'stack_name': 'my_stack', 'creation_time': '2013-08-04T20:57:55Z', 'updated_time': '2013-08-04T20:57:55Z', 'stack_status': 'CREATE_COMPLETE'})
    columns = ['ID', 'Stack Name', 'Stack Status', 'Creation Time', 'Updated Time']

    def _setUp(self, cmd, action, action_name=None):
        self.cmd = cmd
        self.action = action
        self.action_name = action_name
        self.mock_client.stacks.get.return_value = self.stack

    def _test_stack_action(self, get_call_count=1):
        arglist = ['my_stack']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, rows = self.cmd.take_action(parsed_args)
        self.action.assert_called_once_with('my_stack')
        self.mock_client.stacks.get.assert_called_with('my_stack')
        self.assertEqual(get_call_count, self.mock_client.stacks.get.call_count)
        self.assertEqual(self.columns, columns)
        self.assertEqual(1, len(rows))

    def _test_stack_action_multi(self, get_call_count=2):
        arglist = ['my_stack1', 'my_stack2']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, rows = self.cmd.take_action(parsed_args)
        self.assertEqual(2, self.action.call_count)
        self.assertEqual(get_call_count, self.mock_client.stacks.get.call_count)
        self.action.assert_called_with('my_stack2')
        self.mock_client.stacks.get.assert_called_with('my_stack2')
        self.assertEqual(self.columns, columns)
        self.assertEqual(2, len(rows))

    @mock.patch('heatclient.common.event_utils.poll_for_events')
    @mock.patch('heatclient.common.event_utils.get_events', return_value=[])
    def _test_stack_action_wait(self, ge, mock_poll):
        arglist = ['my_stack', '--wait']
        mock_poll.return_value = ('%s_COMPLETE' % self.action_name, 'Stack my_stack %s_COMPLETE' % self.action_name)
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, rows = self.cmd.take_action(parsed_args)
        self.action.assert_called_with('my_stack')
        self.mock_client.stacks.get.assert_called_with('my_stack')
        self.assertEqual(self.columns, columns)
        self.assertEqual(1, len(rows))

    @mock.patch('heatclient.common.event_utils.poll_for_events')
    @mock.patch('heatclient.common.event_utils.get_events', return_value=[])
    def _test_stack_action_wait_error(self, ge, mock_poll):
        arglist = ['my_stack', '--wait']
        mock_poll.return_value = ('%s_FAILED' % self.action_name, 'Error waiting for status from stack my_stack')
        parsed_args = self.check_parser(self.cmd, arglist, [])
        error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertEqual('Error waiting for status from stack my_stack', str(error))

    def _test_stack_action_exception(self):
        self.action.side_effect = heat_exc.HTTPNotFound
        arglist = ['my_stack']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertEqual('Stack not found: my_stack', str(error))