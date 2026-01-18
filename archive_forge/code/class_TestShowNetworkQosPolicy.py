from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_policy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowNetworkQosPolicy(TestQosPolicy):
    _qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
    columns = ('description', 'id', 'is_default', 'name', 'project_id', 'rules', 'shared')
    data = (_qos_policy.description, _qos_policy.id, _qos_policy.is_default, _qos_policy.name, _qos_policy.project_id, network_qos_policy.RulesColumn(_qos_policy.rules), _qos_policy.shared)

    def setUp(self):
        super(TestShowNetworkQosPolicy, self).setUp()
        self.network_client.find_qos_policy = mock.Mock(return_value=self._qos_policy)
        self.cmd = network_qos_policy.ShowNetworkQosPolicy(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_all_options(self):
        arglist = [self._qos_policy.name]
        verifylist = [('policy', self._qos_policy.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_qos_policy.assert_called_once_with(self._qos_policy.name, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(list(self.data), list(data))