from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowNetworkQosDSCPMarking(TestNetworkQosRule):

    def setUp(self):
        super(TestShowNetworkQosDSCPMarking, self).setUp()
        attrs = {'qos_policy_id': self.qos_policy.id, 'type': RULE_TYPE_DSCP_MARKING}
        self.new_rule = network_fakes.FakeNetworkQosRule.create_one_qos_rule(attrs)
        self.qos_policy.rules = [self.new_rule]
        self.columns = ('dscp_mark', 'id', 'project_id', 'qos_policy_id', 'type')
        self.data = (self.new_rule.dscp_mark, self.new_rule.id, self.new_rule.project_id, self.new_rule.qos_policy_id, self.new_rule.type)
        self.network_client.get_qos_dscp_marking_rule = mock.Mock(return_value=self.new_rule)
        self.cmd = network_qos_rule.ShowNetworkQosRule(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_all_options(self):
        arglist = [self.new_rule.qos_policy_id, self.new_rule.id]
        verifylist = [('qos_policy', self.new_rule.qos_policy_id), ('id', self.new_rule.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.get_qos_dscp_marking_rule.assert_called_once_with(self.new_rule.id, self.qos_policy.id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(list(self.data), list(data))