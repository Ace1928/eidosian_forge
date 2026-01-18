import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resources as v1_resources
class TestResourceSignal(TestResource):

    def setUp(self):
        super(TestResourceSignal, self).setUp()
        self.cmd = resource.ResourceSignal(self.app, None)

    def test_resource_signal(self):
        arglist = ['my_stack', 'my_resource']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.resource_client.signal.assert_called_with(**{'stack_id': 'my_stack', 'resource_name': 'my_resource'})

    def test_resource_signal_error(self):
        arglist = ['my_stack', 'my_resource']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.resource_client.signal.side_effect = heat_exc.HTTPNotFound
        error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertEqual('Stack my_stack or resource my_resource not found.', str(error))

    def test_resource_signal_data(self):
        arglist = ['my_stack', 'my_resource', '--data', '{"message":"Content"}']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.resource_client.signal.assert_called_with(**{'data': {'message': 'Content'}, 'stack_id': 'my_stack', 'resource_name': 'my_resource'})

    def test_resource_signal_data_not_json(self):
        arglist = ['my_stack', 'my_resource', '--data', '{']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('Data should be in JSON format', str(error))

    def test_resource_signal_data_and_file_error(self):
        arglist = ['my_stack', 'my_resource', '--data', '{}', '--data-file', 'file']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertEqual('Should only specify one of data or data-file', str(error))

    @mock.patch('urllib.request.urlopen')
    def test_resource_signal_file(self, urlopen):
        data = mock.Mock()
        data.read.side_effect = ['{"message":"Content"}']
        urlopen.return_value = data
        arglist = ['my_stack', 'my_resource', '--data-file', 'test_file']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.resource_client.signal.assert_called_with(**{'data': {'message': 'Content'}, 'stack_id': 'my_stack', 'resource_name': 'my_resource'})