from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
class TestConfigurationCreate(TestConfigurations):
    values = ('2015-05-16T10:24:28', 'mysql', '5.6', '5.7.29', '', 'c-123', 'test_config', '2015-05-16T10:24:29', '{"max_connections": 5}')

    def setUp(self):
        super(TestConfigurationCreate, self).setUp()
        self.cmd = database_configurations.CreateDatabaseConfiguration(self.app, None)
        self.data = self.fake_configurations.get_configurations_c_123()
        self.configuration_client.create.return_value = self.data
        self.columns = ('created', 'datastore_name', 'datastore_version_name', 'datastore_version_number', 'description', 'id', 'name', 'updated', 'values')

    def test_configuration_create_return_value(self):
        args = ['c-123', '{"max_connections": 5}', '--description', 'test_config', '--datastore', 'mysql', '--datastore-version', '5.6']
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.values, data)

    def test_configuration_create(self):
        args = ['cgroup1', '{"param1": 1, "param2": 2}']
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.configuration_client.create.assert_called_with('cgroup1', '{"param1": 1, "param2": 2}', description=None, datastore=None, datastore_version=None, datastore_version_number=None)

    def test_configuration_create_with_optional_args(self):
        args = ['cgroup2', '{"param3": 3, "param4": 4}', '--description', 'cgroup 2', '--datastore', 'mysql', '--datastore-version', '5.6']
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.configuration_client.create.assert_called_with('cgroup2', '{"param3": 3, "param4": 4}', description='cgroup 2', datastore='mysql', datastore_version='5.6', datastore_version_number=None)