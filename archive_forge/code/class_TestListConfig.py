from unittest import mock
from osc_lib import exceptions as exc
import yaml
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_config
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
class TestListConfig(TestConfig):

    def setUp(self):
        super(TestListConfig, self).setUp()
        self.cmd = software_config.ListConfig(self.app, None)
        self.mock_client.software_configs.list.return_value = [software_configs.SoftwareConfig(None, {})]

    def test_config_list(self):
        arglist = []
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.mock_client.software_configs.list.assert_called_once_with()

    def test_config_list_limit(self):
        arglist = ['--limit', '3']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.mock_client.software_configs.list.assert_called_with(limit='3')

    def test_config_list_marker(self):
        arglist = ['--marker', 'id123']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.mock_client.software_configs.list.assert_called_with(marker='id123')