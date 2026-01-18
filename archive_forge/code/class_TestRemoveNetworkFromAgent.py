from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network_agent
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestRemoveNetworkFromAgent(TestNetworkAgent):
    net = network_fakes.create_one_network()
    agent = network_fakes.create_one_network_agent()

    def setUp(self):
        super(TestRemoveNetworkFromAgent, self).setUp()
        self.network_client.get_agent = mock.Mock(return_value=self.agent)
        self.network_client.find_network = mock.Mock(return_value=self.net)
        self.network_client.name = self.network_client.find_network.name
        self.cmd = network_agent.RemoveNetworkFromAgent(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_network_agents_list_routers_no_arg(self):
        arglist = ['--routers']
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_network_from_dhcp_agent(self):
        arglist = ['--dhcp', self.agent.id, self.net.id]
        verifylist = [('dhcp', True), ('agent_id', self.agent.id), ('network', self.net.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.network_client.remove_dhcp_agent_from_network.assert_called_once_with(self.agent, self.net)