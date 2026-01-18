from unittest import mock
from oslo_serialization import jsonutils as json
from saharaclient.api import plugins as api_plugins
from saharaclient.osc.v1 import plugins as osc_plugins
from saharaclient.tests.unit.osc.v1 import fakes
class TestShowPlugin(TestPlugins):

    def setUp(self):
        super(TestShowPlugin, self).setUp()
        self.plugins_mock.get.return_value = api_plugins.Plugin(None, PLUGIN_INFO)
        self.plugins_mock.get_version_details.return_value = api_plugins.Plugin(None, PLUGIN_INFO)
        self.cmd = osc_plugins.ShowPlugin(self.app, None)

    def test_plugin_show(self):
        arglist = ['fake']
        verifylist = [('plugin', 'fake')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.plugins_mock.get.assert_called_once_with('fake')
        expected_columns = ('Description', 'Name', 'Title', 'Versions', '', 'Plugin version 0.1: enabled', 'Plugin: enabled')
        self.assertEqual(expected_columns, columns)
        expected_data = ('Plugin for tests', 'fake', 'Fake Plugin', '0.1, 0.2', '', True, True)
        self.assertEqual(expected_data, data)

    def test_plugin_version_show(self):
        arglist = ['fake', '--plugin-version', '0.1']
        verifylist = [('plugin', 'fake'), ('plugin_version', '0.1')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.plugins_mock.get_version_details.assert_called_once_with('fake', '0.1')
        expected_columns = ('Description', 'Name', 'Required image tags', 'Title', '', 'Plugin version 0.1: enabled', 'Plugin: enabled', '', 'Service:', '', 'HDFS', 'MapReduce')
        self.assertEqual(expected_columns, columns)
        expected_data = ('Plugin for tests', 'fake', '0.1, fake', 'Fake Plugin', '', True, True, '', 'Available processes:', '', 'datanode, namenode', 'jobtracker, tasktracker')
        self.assertEqual(expected_data, data)