from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.osc.v2 import share_network_subnets as osc_share_subnets
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareNetworkSubnetDelete(TestShareNetworkSubnet):

    def setUp(self):
        super(TestShareNetworkSubnetDelete, self).setUp()
        self.share_network = manila_fakes.FakeShareNetwork.create_one_share_network()
        self.share_networks_mock.get.return_value = self.share_network
        self.share_network_subnets = manila_fakes.FakeShareNetworkSubnet.create_share_network_subnets()
        self.cmd = osc_share_subnets.DeleteShareNetworkSubnet(self.app, None)

    def test_share_network_subnet_delete_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_network_subnets_delete(self):
        arglist = [self.share_network.id, self.share_network_subnets[0].id, self.share_network_subnets[1].id]
        verifylist = [('share_network', self.share_network.id), ('share_network_subnet', [self.share_network_subnets[0].id, self.share_network_subnets[1].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertEqual(self.share_subnets_mock.delete.call_count, len(self.share_network_subnets))
        self.assertIsNone(result)

    def test_share_network_subnet_delete_exception(self):
        arglist = [self.share_network.id, self.share_network_subnets[0].id]
        verifylist = [('share_network', self.share_network.id), ('share_network_subnet', [self.share_network_subnets[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.share_subnets_mock.delete.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)