from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
class TestUpdateDatabaseConfiguration(TestConfigurations):

    def setUp(self):
        super(TestUpdateDatabaseConfiguration, self).setUp()
        self.cmd = database_configurations.UpdateDatabaseConfiguration(self.app, None)

    def test_set_database_configuration_parameter(self):
        args = ['config_group_id', '{"param1": 1, "param2": 2}', '--name', 'new_name']
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.configuration_client.update.assert_called_once_with('config_group_id', '{"param1": 1, "param2": 2}', name='new_name', description=None)