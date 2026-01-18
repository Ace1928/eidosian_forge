from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListNetworkQosRule(TestNetworkQosRule):

    def setUp(self):
        super(TestListNetworkQosRule, self).setUp()
        attrs = {'qos_policy_id': self.qos_policy.id, 'type': RULE_TYPE_MINIMUM_BANDWIDTH}
        self.new_rule_min_bw = network_fakes.FakeNetworkQosRule.create_one_qos_rule(attrs=attrs)
        attrs['type'] = RULE_TYPE_MINIMUM_PACKET_RATE
        self.new_rule_min_pps = network_fakes.FakeNetworkQosRule.create_one_qos_rule(attrs=attrs)
        attrs['type'] = RULE_TYPE_DSCP_MARKING
        self.new_rule_dscp_mark = network_fakes.FakeNetworkQosRule.create_one_qos_rule(attrs=attrs)
        attrs['type'] = RULE_TYPE_BANDWIDTH_LIMIT
        self.new_rule_max_bw = network_fakes.FakeNetworkQosRule.create_one_qos_rule(attrs=attrs)
        self.qos_policy.rules = [self.new_rule_min_bw, self.new_rule_min_pps, self.new_rule_dscp_mark, self.new_rule_max_bw]
        self.network_client.find_qos_minimum_bandwidth_rule = mock.Mock(return_value=self.new_rule_min_bw)
        self.network_client.find_qos_minimum_packet_rate_rule = mock.Mock(return_value=self.new_rule_min_pps)
        self.network_client.find_qos_dscp_marking_rule = mock.Mock(return_value=self.new_rule_dscp_mark)
        self.network_client.find_qos_bandwidth_limit_rule = mock.Mock(return_value=self.new_rule_max_bw)
        self.columns = ('ID', 'QoS Policy ID', 'Type', 'Max Kbps', 'Max Burst Kbits', 'Min Kbps', 'Min Kpps', 'DSCP mark', 'Direction')
        self.data = []
        for index in range(len(self.qos_policy.rules)):
            self.data.append((self.qos_policy.rules[index].id, self.qos_policy.rules[index].qos_policy_id, self.qos_policy.rules[index].type, getattr(self.qos_policy.rules[index], 'max_kbps', ''), getattr(self.qos_policy.rules[index], 'max_burst_kbps', ''), getattr(self.qos_policy.rules[index], 'min_kbps', ''), getattr(self.qos_policy.rules[index], 'min_kpps', ''), getattr(self.qos_policy.rules[index], 'dscp_mark', ''), getattr(self.qos_policy.rules[index], 'direction', '')))
        self.cmd = network_qos_rule.ListNetworkQosRule(self.app, self.namespace)

    def test_qos_rule_list(self):
        arglist = [self.qos_policy.id]
        verifylist = [('qos_policy', self.qos_policy.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_qos_policy.assert_called_once_with(self.qos_policy.id, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        list_data = list(data)
        self.assertEqual(len(self.data), len(list_data))
        for index in range(len(list_data)):
            self.assertEqual(self.data[index], list_data[index])