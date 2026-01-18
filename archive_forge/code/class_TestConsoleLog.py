from unittest import mock
from openstackclient.compute.v2 import console
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils
class TestConsoleLog(compute_fakes.TestComputev2):
    _server = compute_fakes.create_one_server()

    def setUp(self):
        super(TestConsoleLog, self).setUp()
        self.compute_sdk_client.find_server.return_value = self._server
        self.cmd = console.ShowConsoleLog(self.app, None)

    def test_show_no_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show(self):
        arglist = ['fake_server']
        verifylist = [('server', 'fake_server')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        output = {'output': '1st line\n2nd line\n'}
        self.compute_sdk_client.get_server_console_output.return_value = output
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_server.assert_called_with(name_or_id='fake_server', ignore_missing=False)
        self.compute_sdk_client.get_server_console_output.assert_called_with(self._server.id, length=None)
        stdout = self.app.stdout.content
        self.assertEqual(stdout[0], output['output'])

    def test_show_lines(self):
        arglist = ['fake_server', '--lines', '15']
        verifylist = [('server', 'fake_server'), ('lines', 15)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        output = {'output': '1st line\n2nd line'}
        self.compute_sdk_client.get_server_console_output.return_value = output
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_server.assert_called_with(name_or_id='fake_server', ignore_missing=False)
        self.compute_sdk_client.get_server_console_output.assert_called_with(self._server.id, length=15)