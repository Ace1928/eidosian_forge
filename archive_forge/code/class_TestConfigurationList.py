from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
class TestConfigurationList(TestConfigurations):
    defaults = {'limit': None, 'marker': None}
    columns = database_configurations.ListDatabaseConfigurations.columns
    values = ('c-123', 'test_config', '', 'mysql', '5.6', '5.7.29')

    def setUp(self):
        super(TestConfigurationList, self).setUp()
        self.cmd = database_configurations.ListDatabaseConfigurations(self.app, None)
        data = [self.fake_configurations.get_configurations_c_123()]
        self.configuration_client.list.return_value = common.Paginated(data)

    def test_configuration_list_defaults(self):
        parsed_args = self.check_parser(self.cmd, [], [])
        columns, data = self.cmd.take_action(parsed_args)
        self.configuration_client.list.assert_called_once_with(**self.defaults)
        self.assertEqual(self.columns, columns)
        self.assertEqual([tuple(self.values)], data)