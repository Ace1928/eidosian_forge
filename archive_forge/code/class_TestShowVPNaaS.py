import testtools
from osc_lib import exceptions
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
class TestShowVPNaaS(test_fakes.TestNeutronClientOSCV2):

    def test_show_filtered_by_id_or_name(self):
        target = self.resource['id']

        def _mock_vpnaas(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_vpn_endpoint_group.side_effect = _mock_vpnaas
        self.networkclient.find_vpn_ipsec_site_connection.side_effect = _mock_vpnaas
        self.networkclient.find_vpn_ike_policy.side_effect = _mock_vpnaas
        self.networkclient.find_vpn_ipsec_policy.side_effect = _mock_vpnaas
        self.networkclient.find_vpn_service.side_effect = _mock_vpnaas
        arglist = [target]
        verifylist = [(self.res, target)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target)
        self.assertEqual(self.ordered_headers, headers)
        self.assertItemEqual(self.ordered_data, data)