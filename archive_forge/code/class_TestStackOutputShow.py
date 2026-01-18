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
class TestStackOutputShow(TestStack):
    outputs = [{'output_key': 'output1', 'output_value': 'value1'}, {'output_key': 'output2', 'output_value': 'value2', 'output_error': 'error'}]
    response = {'outputs': outputs, 'stack_name': 'my_stack'}

    def setUp(self):
        super(TestStackOutputShow, self).setUp()
        self.cmd = stack.OutputShowStack(self.app, None)
        self.stack_client.get.return_value = stacks.Stack(None, self.response)

    def test_stack_output_show_no_output(self):
        arglist = ['my_stack']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertEqual('Either <OUTPUT NAME> or --all must be specified.', str(error))

    def test_stack_output_show_output_and_all(self):
        arglist = ['my_stack', 'output1', '--all']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertEqual('Cannot specify both <OUTPUT NAME> and --all.', str(error))

    def test_stack_output_show_all(self):
        arglist = ['my_stack', '--all']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, outputs = self.cmd.take_action(parsed_args)
        self.stack_client.get.assert_called_with('my_stack')
        self.assertEqual(['output1', 'output2'], columns)

    def test_stack_output_show_output(self):
        arglist = ['my_stack', 'output1']
        self.stack_client.output_show.return_value = {'output': self.outputs[0]}
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, outputs = self.cmd.take_action(parsed_args)
        self.stack_client.output_show.assert_called_with('my_stack', 'output1')
        self.assertEqual(('output_key', 'output_value'), columns)
        self.assertEqual(('output1', 'value1'), outputs)

    def test_stack_output_show_not_found(self):
        arglist = ['my_stack', '--all']
        self.stack_client.get.side_effect = heat_exc.HTTPNotFound
        parsed_args = self.check_parser(self.cmd, arglist, [])
        error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertEqual('Stack not found: my_stack', str(error))

    def test_stack_output_show_output_error(self):
        arglist = ['my_stack', 'output2']
        self.stack_client.output_show.return_value = {'output': self.outputs[1]}
        parsed_args = self.check_parser(self.cmd, arglist, [])
        error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertEqual('Output error: error', str(error))
        self.stack_client.output_show.assert_called_with('my_stack', 'output2')

    def test_stack_output_show_bad_output(self):
        arglist = ['my_stack', 'output3']
        self.stack_client.output_show.side_effect = heat_exc.HTTPNotFound
        self.stack_client.get.side_effect = heat_exc.HTTPNotFound
        parsed_args = self.check_parser(self.cmd, arglist, [])
        error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertEqual('Stack my_stack or output output3 not found.', str(error))
        self.stack_client.output_show.assert_called_with('my_stack', 'output3')

    def test_stack_output_show_old_api(self):
        arglist = ['my_stack', 'output1']
        self.stack_client.output_show.side_effect = heat_exc.HTTPNotFound
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, outputs = self.cmd.take_action(parsed_args)
        self.stack_client.get.assert_called_with('my_stack')
        self.assertEqual(('output_key', 'output_value'), columns)
        self.assertEqual(('output1', 'value1'), outputs)