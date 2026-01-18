from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.osc.v2 import share_network_subnets as osc_share_subnets
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareNetworkSubnetSet(TestShareNetworkSubnet):

    def setUp(self):
        super(TestShareNetworkSubnetSet, self).setUp()
        self.share_network = manila_fakes.FakeShareNetwork.create_one_share_network()
        self.share_networks_mock.get.return_value = self.share_network
        self.share_network_subnet = manila_fakes.FakeShareNetworkSubnet.create_one_share_subnet()
        self.cmd = osc_share_subnets.SetShareNetworkSubnet(self.app, None)

    def test_set_share_network_subnet_property(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.78')
        arglist = [self.share_network.id, self.share_network_subnet.id, '--property', 'Zorilla=manila', '--property', 'test=my_test']
        verifylist = [('share_network', self.share_network.id), ('share_network_subnet', self.share_network_subnet.id), ('property', {'Zorilla': 'manila', 'test': 'my_test'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.share_subnets_mock.set_metadata.assert_called_once_with(self.share_network.id, {'Zorilla': 'manila', 'test': 'my_test'}, subresource=self.share_network_subnet.id)

    def test_set_share_network_subnet_property_exception(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.78')
        arglist = [self.share_network.id, self.share_network_subnet.id, '--property', 'key=1']
        verifylist = [('share_network', self.share_network.id), ('share_network_subnet', self.share_network_subnet.id), ('property', {'key': '1'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.share_subnets_mock.set_metadata.assert_called_once_with(self.share_network.id, {'key': '1'}, subresource=self.share_network_subnet.id)
        self.share_subnets_mock.set_metadata.side_effect = exceptions.BadRequest
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)