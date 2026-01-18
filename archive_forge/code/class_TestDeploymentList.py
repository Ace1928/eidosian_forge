import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_deployment
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
from heatclient.v1 import software_deployments
class TestDeploymentList(TestDeployment):
    columns = ['id', 'config_id', 'server_id', 'action', 'status']
    data = {'software_deployments': [{'status': 'COMPLETE', 'server_id': 'ec14c864-096e-4e27-bb8a-2c2b4dc6f3f5', 'config_id': '8da95794-2ad9-4979-8ae5-739ce314c5cd', 'output_values': {'deploy_stdout': 'Writing to /tmp/barmy Written to /tmp/barmy', 'deploy_stderr': '+ echo Writing to /tmp/barmy\n+ echo fu\n+ cat /tmp/barmy\n+ echo -n The file /tmp/barmycontains for server ec14c864-096e-4e27-bb8a-2c2b4dc6f3f5 during CREATE\n+echo Output to stderr\nOutput to stderr\n', 'deploy_status_code': 0, 'result': 'The file /tmp/barmy contains fu for server ec14c864-096e-4e27-bb8a-2c2b4dc6f3f5 during CREATE'}, 'input_values': None, 'action': 'CREATE', 'status_reason': 'Outputs received', 'id': 'ef422fa5-719a-419e-a10c-72e3a367b0b8', 'creation_time': '2015-01-31T15:12:36Z', 'updated_time': '2015-01-31T15:18:21Z'}]}

    def setUp(self):
        super(TestDeploymentList, self).setUp()
        self.cmd = software_deployment.ListDeployment(self.app, None)
        self.sd_client.list = mock.MagicMock(return_value=[self.data])

    def test_deployment_list(self):
        arglist = []
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.sd_client.list.assert_called_with()
        self.assertEqual(self.columns, columns)

    def test_deployment_list_server(self):
        kwargs = {}
        kwargs['server_id'] = 'ec14c864-096e-4e27-bb8a-2c2b4dc6f3f5'
        arglist = ['--server', 'ec14c864-096e-4e27-bb8a-2c2b4dc6f3f5']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.sd_client.list.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)

    def test_deployment_list_long(self):
        kwargs = {}
        cols = ['id', 'config_id', 'server_id', 'action', 'status', 'creation_time', 'status_reason']
        arglist = ['--long']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.sd_client.list.assert_called_with(**kwargs)
        self.assertEqual(cols, columns)