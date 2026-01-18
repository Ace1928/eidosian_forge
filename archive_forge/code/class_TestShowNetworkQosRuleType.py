from unittest import mock
from openstackclient.network.v2 import network_qos_rule_type as _qos_rule_type
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowNetworkQosRuleType(TestNetworkQosRuleType):
    attrs = {'drivers': [{'name': 'driver 1', 'supported_parameters': []}]}
    qos_rule_type = network_fakes.FakeNetworkQosRuleType.create_one_qos_rule_type(attrs)
    columns = ('drivers', 'rule_type_name')
    data = [qos_rule_type.drivers, qos_rule_type.type]

    def setUp(self):
        super(TestShowNetworkQosRuleType, self).setUp()
        self.network_client.get_qos_rule_type = mock.Mock(return_value=self.qos_rule_type)
        self.cmd = _qos_rule_type.ShowNetworkQosRuleType(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_all_options(self):
        arglist = [self.qos_rule_type.type]
        verifylist = [('rule_type', self.qos_rule_type.type)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.get_qos_rule_type.assert_called_once_with(self.qos_rule_type.type)
        self.assertEqual(self.columns, columns)
        self.assertEqual(list(self.data), list(data))