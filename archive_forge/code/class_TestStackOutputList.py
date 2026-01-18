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
class TestStackOutputList(TestStack):
    response = {'outputs': [{'output_key': 'key1', 'description': 'desc1'}, {'output_key': 'key2', 'description': 'desc2'}]}
    stack_response = {'stack_name': 'my_stack', 'outputs': response['outputs']}

    def setUp(self):
        super(TestStackOutputList, self).setUp()
        self.cmd = stack.OutputListStack(self.app, None)
        self.stack_client.get = mock.MagicMock(return_value=stacks.Stack(None, self.response))

    def test_stack_output_list(self):
        arglist = ['my_stack']
        self.stack_client.output_list.return_value = self.response
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(['output_key', 'description'], columns)
        self.stack_client.output_list.assert_called_with('my_stack')

    def test_stack_output_list_not_found(self):
        arglist = ['my_stack']
        self.stack_client.output_list.side_effect = heat_exc.HTTPNotFound
        self.stack_client.get.side_effect = heat_exc.HTTPNotFound
        parsed_args = self.check_parser(self.cmd, arglist, [])
        error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertEqual('Stack not found: my_stack', str(error))

    def test_stack_output_list_old_api(self):
        arglist = ['my_stack']
        self.stack_client.output_list.side_effect = heat_exc.HTTPNotFound
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, outputs = self.cmd.take_action(parsed_args)
        self.stack_client.get.assert_called_with('my_stack')
        self.assertEqual(['output_key', 'description'], columns)