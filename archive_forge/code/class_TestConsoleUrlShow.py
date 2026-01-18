from unittest import mock
from openstackclient.compute.v2 import console
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils
class TestConsoleUrlShow(compute_fakes.TestComputev2):
    _server = compute_fakes.create_one_server()

    def setUp(self):
        super(TestConsoleUrlShow, self).setUp()
        self.compute_sdk_client.find_server.return_value = self._server
        fake_console_data = {'url': 'http://localhost', 'protocol': 'fake_protocol', 'type': 'fake_type'}
        self.compute_sdk_client.create_console = mock.Mock(return_value=fake_console_data)
        self.columns = ('protocol', 'type', 'url')
        self.data = (fake_console_data['protocol'], fake_console_data['type'], fake_console_data['url'])
        self.cmd = console.ShowConsoleURL(self.app, None)

    def test_console_url_show_by_default(self):
        arglist = ['foo_vm']
        verifylist = [('url_type', 'novnc'), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_console.assert_called_once_with(self._server.id, console_type='novnc')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_console_url_show_with_novnc(self):
        arglist = ['--novnc', 'foo_vm']
        verifylist = [('url_type', 'novnc'), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_console.assert_called_once_with(self._server.id, console_type='novnc')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_console_url_show_with_xvpvnc(self):
        arglist = ['--xvpvnc', 'foo_vm']
        verifylist = [('url_type', 'xvpvnc'), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_console.assert_called_once_with(self._server.id, console_type='xvpvnc')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_console_url_show_with_spice(self):
        arglist = ['--spice', 'foo_vm']
        verifylist = [('url_type', 'spice-html5'), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_console.assert_called_once_with(self._server.id, console_type='spice-html5')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_console_url_show_with_rdp(self):
        arglist = ['--rdp', 'foo_vm']
        verifylist = [('url_type', 'rdp-html5'), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_console.assert_called_once_with(self._server.id, console_type='rdp-html5')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_console_url_show_with_serial(self):
        arglist = ['--serial', 'foo_vm']
        verifylist = [('url_type', 'serial'), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_console.assert_called_once_with(self._server.id, console_type='serial')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_console_url_show_with_mks(self):
        arglist = ['--mks', 'foo_vm']
        verifylist = [('url_type', 'webmks'), ('server', 'foo_vm')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_console.assert_called_once_with(self._server.id, console_type='webmks')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)