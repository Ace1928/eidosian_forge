from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_meter_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowMeterRule(TestMeterRule):
    new_rule = network_fakes.FakeNetworkMeterRule.create_one_rule()
    columns = ('destination_ip_prefix', 'direction', 'excluded', 'id', 'metering_label_id', 'project_id', 'remote_ip_prefix', 'source_ip_prefix')
    data = (new_rule.destination_ip_prefix, new_rule.direction, new_rule.excluded, new_rule.id, new_rule.metering_label_id, new_rule.project_id, new_rule.remote_ip_prefix, new_rule.source_ip_prefix)

    def setUp(self):
        super(TestShowMeterRule, self).setUp()
        self.cmd = network_meter_rule.ShowMeterRule(self.app, self.namespace)
        self.network_client.find_metering_label_rule = mock.Mock(return_value=self.new_rule)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_label_rule_show_option(self):
        arglist = [self.new_rule.id]
        verifylist = [('meter_rule_id', self.new_rule.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_metering_label_rule.assert_called_with(self.new_rule.id, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)