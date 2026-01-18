from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def _set_min_kpps(self, min_kpps=None):
    if min_kpps:
        previous_min_kpps = self.new_rule.min_kpps
        self.new_rule.min_kpps = min_kpps
    arglist = ['--min-kpps', str(self.new_rule.min_kpps), self.new_rule.qos_policy_id, self.new_rule.id]
    verifylist = [('min_kpps', self.new_rule.min_kpps), ('qos_policy', self.new_rule.qos_policy_id), ('id', self.new_rule.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'min_kpps': self.new_rule.min_kpps}
    self.network_client.update_qos_minimum_packet_rate_rule.assert_called_with(self.new_rule, self.qos_policy.id, **attrs)
    self.assertIsNone(result)
    if min_kpps:
        self.new_rule.min_kpps = previous_min_kpps