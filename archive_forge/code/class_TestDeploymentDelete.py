import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_deployment
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
from heatclient.v1 import software_deployments
class TestDeploymentDelete(TestDeployment):

    def setUp(self):
        super(TestDeploymentDelete, self).setUp()
        self.cmd = software_deployment.DeleteDeployment(self.app, None)

    def test_deployment_delete_success(self):
        arglist = ['test_deployment']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.sd_client.delete.assert_called_with(deployment_id='test_deployment')

    def test_deployment_delete_multiple(self):
        arglist = ['test_deployment', 'test_deployment2']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.sd_client.delete.assert_has_calls([mock.call(deployment_id='test_deployment'), mock.call(deployment_id='test_deployment2')])

    def test_deployment_delete_not_found(self):
        arglist = ['test_deployment', 'test_deployment2']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.sd_client.delete.side_effect = heat_exc.HTTPNotFound()
        error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('Unable to delete 2 of the 2 deployments.', str(error))

    def test_deployment_config_delete_failed(self):
        arglist = ['test_deployment']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.config_client.delete.side_effect = heat_exc.HTTPNotFound()
        self.assertIsNone(self.cmd.take_action(parsed_args))