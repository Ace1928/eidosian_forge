from unittest import mock
from osc_lib import exceptions as exc
import yaml
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_config
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
class TestDeleteConfig(TestConfig):

    def setUp(self):
        super(TestDeleteConfig, self).setUp()
        self.cmd = software_config.DeleteConfig(self.app, None)
        self.mock_delete = self.mock_client.software_configs.delete

    def test_config_delete(self):
        arglist = ['id_123']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.mock_delete.assert_called_with(config_id='id_123')

    def test_config_delete_multi(self):
        arglist = ['id_123', 'id_456']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.mock_delete.assert_has_calls([mock.call(config_id='id_123'), mock.call(config_id='id_456')])

    def test_config_delete_not_found(self):
        arglist = ['id_123', 'id_456', 'id_789']
        self.mock_client.software_configs.delete.side_effect = [None, heat_exc.HTTPNotFound, None]
        parsed_args = self.check_parser(self.cmd, arglist, [])
        error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.mock_delete.assert_has_calls([mock.call(config_id='id_123'), mock.call(config_id='id_456'), mock.call(config_id='id_789')])
        self.assertEqual('Unable to delete 1 of the 3 software configs.', str(error))