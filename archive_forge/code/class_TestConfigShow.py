from unittest import mock
from osc_lib import exceptions as exc
import yaml
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_config
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
class TestConfigShow(TestConfig):
    columns = ('id', 'name', 'group', 'config', 'inputs', 'outputs', 'options', 'creation_time')
    data = ('96dfee3f-27b7-42ae-a03e-966226871ae6', 'test', 'Heat::Ungrouped', '', [], [], {}, '2015-12-09T11:55:06')
    response = dict(zip(columns, data))

    def setUp(self):
        super(TestConfigShow, self).setUp()
        self.cmd = software_config.ShowConfig(self.app, None)
        self.mock_client.software_configs.get.return_value = software_configs.SoftwareConfig(None, self.response)

    def test_config_show(self):
        arglist = ['96dfee3f-27b7-42ae-a03e-966226871ae6']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.mock_client.software_configs.get.assert_called_with(**{'config_id': '96dfee3f-27b7-42ae-a03e-966226871ae6'})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_config_show_config_only(self):
        arglist = ['--config-only', '96dfee3f-27b7-42ae-a03e-966226871ae6']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.mock_client.software_configs.get.assert_called_with(**{'config_id': '96dfee3f-27b7-42ae-a03e-966226871ae6'})
        self.assertIsNone(columns)
        self.assertIsNone(data)

    def test_config_show_not_found(self):
        arglist = ['96dfee3f-27b7-42ae-a03e-966226871ae6']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.mock_client.software_configs.get.side_effect = heat_exc.HTTPNotFound()
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)