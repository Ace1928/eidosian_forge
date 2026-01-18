from unittest import mock
from oslo_serialization import jsonutils as json
from saharaclient.api import plugins as api_plugins
from saharaclient.osc.v1 import plugins as osc_plugins
from saharaclient.tests.unit.osc.v1 import fakes
class TestListPlugins(TestPlugins):

    def setUp(self):
        super(TestListPlugins, self).setUp()
        self.plugins_mock.list.return_value = [api_plugins.Plugin(None, PLUGIN_INFO)]
        self.cmd = osc_plugins.ListPlugins(self.app, None)

    def test_plugins_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        expected_columns = ['Name', 'Versions']
        self.assertEqual(expected_columns, columns)
        expected_data = [('fake', '0.1, 0.2')]
        self.assertEqual(expected_data, list(data))

    def test_plugins_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        expected_columns = ['Name', 'Title', 'Versions', 'Description']
        self.assertEqual(expected_columns, columns)
        expected_data = [('fake', 'Fake Plugin', '0.1, 0.2', 'Plugin for tests')]
        self.assertEqual(expected_data, list(data))