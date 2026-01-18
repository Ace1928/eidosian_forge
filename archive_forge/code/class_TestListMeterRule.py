from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_meter_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListMeterRule(TestMeterRule):
    rule_list = network_fakes.FakeNetworkMeterRule.create_meter_rule(count=2)
    columns = ('ID', 'Excluded', 'Direction', 'Remote IP Prefix', 'Source IP Prefix', 'Destination IP Prefix')
    data = []
    for rule in rule_list:
        data.append((rule.id, rule.excluded, rule.direction, rule.remote_ip_prefix, rule.source_ip_prefix, rule.destination_ip_prefix))

    def setUp(self):
        super(TestListMeterRule, self).setUp()
        self.network_client.metering_label_rules = mock.Mock(return_value=self.rule_list)
        self.cmd = network_meter_rule.ListMeterRule(self.app, self.namespace)

    def test_rule_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.metering_label_rules.assert_called_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))