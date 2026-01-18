from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_meter_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestCreateMeterRule(TestMeterRule):
    project = identity_fakes_v3.FakeProject.create_one_project()
    domain = identity_fakes_v3.FakeDomain.create_one_domain()
    new_rule = network_fakes.FakeNetworkMeterRule.create_one_rule()
    columns = ('destination_ip_prefix', 'direction', 'excluded', 'id', 'metering_label_id', 'project_id', 'remote_ip_prefix', 'source_ip_prefix')
    data = (new_rule.destination_ip_prefix, new_rule.direction, new_rule.excluded, new_rule.id, new_rule.metering_label_id, new_rule.project_id, new_rule.remote_ip_prefix, new_rule.source_ip_prefix)

    def setUp(self):
        super(TestCreateMeterRule, self).setUp()
        fake_meter = network_fakes.FakeNetworkMeter.create_one_meter({'id': self.new_rule.metering_label_id})
        self.network_client.create_metering_label_rule = mock.Mock(return_value=self.new_rule)
        self.projects_mock.get.return_value = self.project
        self.cmd = network_meter_rule.CreateMeterRule(self.app, self.namespace)
        self.network_client.find_metering_label = mock.Mock(return_value=fake_meter)

    def test_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_default_options(self):
        arglist = [self.new_rule.metering_label_id, '--remote-ip-prefix', self.new_rule.remote_ip_prefix]
        verifylist = [('meter', self.new_rule.metering_label_id), ('remote_ip_prefix', self.new_rule.remote_ip_prefix)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_metering_label_rule.assert_called_once_with(**{'direction': 'ingress', 'metering_label_id': self.new_rule.metering_label_id, 'remote_ip_prefix': self.new_rule.remote_ip_prefix})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_all_options(self):
        arglist = ['--ingress', '--include', self.new_rule.metering_label_id, '--remote-ip-prefix', self.new_rule.remote_ip_prefix]
        verifylist = [('ingress', True), ('include', True), ('meter', self.new_rule.metering_label_id), ('remote_ip_prefix', self.new_rule.remote_ip_prefix)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_metering_label_rule.assert_called_once_with(**{'direction': self.new_rule.direction, 'excluded': self.new_rule.excluded, 'metering_label_id': self.new_rule.metering_label_id, 'remote_ip_prefix': self.new_rule.remote_ip_prefix})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)