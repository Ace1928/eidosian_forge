import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_deployment
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
from heatclient.v1 import software_deployments
class TestDeploymentShow(TestDeployment):
    get_response = {'software_deployment': {'status': 'IN_PROGRESS', 'server_id': 'ec14c864-096e-4e27-bb8a-2c2b4dc6f3f5', 'config_id': '3d5ec2a8-7004-43b6-a7f6-542bdbe9d434', 'output_values': 'null', 'input_values': 'null', 'action': 'CREATE', 'status_reason': 'Deploy data available', 'id': '06e87bcc-33a2-4bce-aebd-533e698282d3', 'creation_time': '2015-01-31T15:12:36Z', 'updated_time': '2015-01-31T15:18:21Z'}}

    def setUp(self):
        super(TestDeploymentShow, self).setUp()
        self.cmd = software_deployment.ShowDeployment(self.app, None)

    def test_deployment_show(self):
        arglist = ['my_deployment']
        cols = ['id', 'server_id', 'config_id', 'creation_time', 'updated_time', 'status', 'status_reason', 'input_values', 'action']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.sd_client.get.return_value = software_deployments.SoftwareDeployment(None, self.get_response)
        columns, data = self.cmd.take_action(parsed_args)
        self.sd_client.get.assert_called_with(**{'deployment_id': 'my_deployment'})
        self.assertEqual(cols, columns)

    def test_deployment_show_long(self):
        arglist = ['my_deployment', '--long']
        cols = ['id', 'server_id', 'config_id', 'creation_time', 'updated_time', 'status', 'status_reason', 'input_values', 'action', 'output_values']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.sd_client.get.return_value = software_deployments.SoftwareDeployment(None, self.get_response)
        columns, data = self.cmd.take_action(parsed_args)
        self.sd_client.get.assert_called_once_with(**{'deployment_id': 'my_deployment'})
        self.assertEqual(cols, columns)

    def test_deployment_not_found(self):
        arglist = ['my_deployment']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.sd_client.get.side_effect = heat_exc.HTTPNotFound()
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)