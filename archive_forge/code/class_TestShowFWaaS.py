import testtools
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
class TestShowFWaaS(test_fakes.TestNeutronClientOSCV2):

    def test_show_filtered_by_id_or_name(self):
        target = self.resource['id']
        headers, data = (None, None)

        def _mock_fwaas(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_firewall_policy.side_effect = _mock_fwaas
        self.networkclient.find_firewall_group.side_effect = _mock_fwaas
        self.networkclient.find_firewall_rule.side_effect = _mock_fwaas
        arglist = [target]
        verifylist = [(self.res, target)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target)
        self.assertEqual(self.ordered_headers, headers)