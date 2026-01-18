from unittest import mock
from neutronclient.osc.v2.subnet_onboard import subnet_onboard
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
class TestNetworkOnboardSubnets(test_fakes.TestNeutronClientOSCV2):

    def setUp(self):
        super(TestNetworkOnboardSubnets, self).setUp()
        mock.patch('neutronclient.osc.v2.subnet_onboard.subnet_onboard._get_id', new=_get_id).start()
        self.network_id = 'my_network_id'
        self.subnetpool_id = 'my_subnetpool_id'
        self.cmd = subnet_onboard.NetworkOnboardSubnets(self.app, self.namespace)

    def test_options(self):
        arglist = [self.network_id, self.subnetpool_id]
        verifylist = [('network', self.network_id), ('subnetpool', self.subnetpool_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.neutronclient.onboard_network_subnets.assert_called_once_with(self.subnetpool_id, {'network_id': self.network_id})