import argparse
from unittest import mock
import openstack
from osc_lib import exceptions
from openstackclient.network import common
from openstackclient.tests.unit import utils
class TestNetworkAndCompute(utils.TestCommand):

    def setUp(self):
        super().setUp()
        self.namespace = argparse.Namespace()
        self.app.client_manager.network = mock.Mock()
        self.network_client = self.app.client_manager.network
        self.network_client.network_action = mock.Mock(return_value='take_action_network')
        self.app.client_manager.compute = mock.Mock()
        self.compute_client = self.app.client_manager.compute
        self.compute_client.compute_action = mock.Mock(return_value='take_action_compute')
        self.cmd = FakeNetworkAndComputeCommand(self.app, self.namespace)

    def test_take_action_network(self):
        arglist = ['common', 'network']
        verifylist = [('common', 'common'), ('network', 'network')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.network_action.assert_called_with(parsed_args)
        self.assertEqual('take_action_network', result)

    def test_take_action_compute(self):
        arglist = ['common', 'compute']
        verifylist = [('common', 'common'), ('compute', 'compute')]
        self.app.client_manager.network_endpoint_enabled = False
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.compute_client.compute_action.assert_called_with(parsed_args)
        self.assertEqual('take_action_compute', result)