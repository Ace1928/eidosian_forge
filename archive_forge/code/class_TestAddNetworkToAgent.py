from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network_agent
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestAddNetworkToAgent(TestNetworkAgent):
    net = network_fakes.create_one_network()
    agent = network_fakes.create_one_network_agent()

    def setUp(self):
        super(TestAddNetworkToAgent, self).setUp()
        self.network_client.get_agent = mock.Mock(return_value=self.agent)
        self.network_client.find_network = mock.Mock(return_value=self.net)
        self.network_client.name = self.network_client.find_network.name
        self.cmd = network_agent.AddNetworkToAgent(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_add_network_to_dhcp_agent(self):
        arglist = ['--dhcp', self.agent.id, self.net.id]
        verifylist = [('dhcp', True), ('agent_id', self.agent.id), ('network', self.net.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.network_client.add_dhcp_agent_to_network.assert_called_once_with(self.agent, self.net)