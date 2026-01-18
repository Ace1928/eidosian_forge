from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network_agent
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListNetworkAgent(TestNetworkAgent):
    network_agents = network_fakes.create_network_agents(count=3)
    columns = ('ID', 'Agent Type', 'Host', 'Availability Zone', 'Alive', 'State', 'Binary')
    data = []
    for agent in network_agents:
        data.append((agent.id, agent.agent_type, agent.host, agent.availability_zone, network_agent.AliveColumn(agent.is_alive), network_agent.AdminStateColumn(agent.is_admin_state_up), agent.binary))

    def setUp(self):
        super(TestListNetworkAgent, self).setUp()
        self.network_client.agents = mock.Mock(return_value=self.network_agents)
        _testagent = network_fakes.create_one_network_agent()
        self.network_client.get_agent = mock.Mock(return_value=_testagent)
        self._testnetwork = network_fakes.create_one_network()
        self.network_client.find_network = mock.Mock(return_value=self._testnetwork)
        self.network_client.network_hosting_dhcp_agents = mock.Mock(return_value=self.network_agents)
        self.network_client.get_agent = mock.Mock(return_value=_testagent)
        self._testrouter = network_fakes.FakeRouter.create_one_router()
        self.network_client.find_router = mock.Mock(return_value=self._testrouter)
        self.network_client.routers_hosting_l3_agents = mock.Mock(return_value=self.network_agents)
        self.cmd = network_agent.ListNetworkAgent(self.app, self.namespace)

    def test_network_agents_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.agents.assert_called_once_with(**{})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_agents_list_agent_type(self):
        arglist = ['--agent-type', 'dhcp']
        verifylist = [('agent_type', 'dhcp')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.agents.assert_called_once_with(**{'agent_type': 'DHCP agent'})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_agents_list_host(self):
        arglist = ['--host', self.network_agents[0].host]
        verifylist = [('host', self.network_agents[0].host)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.agents.assert_called_once_with(**{'host': self.network_agents[0].host})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_agents_list_networks(self):
        arglist = ['--network', self._testnetwork.id]
        verifylist = [('network', self._testnetwork.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.network_hosting_dhcp_agents.assert_called_once_with(self._testnetwork)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_agents_list_routers(self):
        arglist = ['--router', self._testrouter.id]
        verifylist = [('router', self._testrouter.id), ('long', False)]
        attrs = {self._testrouter}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.routers_hosting_l3_agents.assert_called_once_with(*attrs)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_agents_list_routers_with_long_option(self):
        arglist = ['--router', self._testrouter.id, '--long']
        verifylist = [('router', self._testrouter.id), ('long', True)]
        attrs = {self._testrouter}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.routers_hosting_l3_agents.assert_called_once_with(*attrs)
        router_agent_columns = self.columns + ('HA State',)
        router_agent_data = [d + ('',) for d in self.data]
        self.assertEqual(router_agent_columns, columns)
        self.assertEqual(len(router_agent_data), len(list(data)))