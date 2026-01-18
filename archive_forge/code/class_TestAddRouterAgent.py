from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network_agent
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestAddRouterAgent(TestNetworkAgent):
    _router = network_fakes.FakeRouter.create_one_router()
    _agent = network_fakes.create_one_network_agent()

    def setUp(self):
        super(TestAddRouterAgent, self).setUp()
        self.network_client.get_agent = mock.Mock(return_value=self._agent)
        self.network_client.find_router = mock.Mock(return_value=self._router)
        self.cmd = network_agent.AddRouterToAgent(self.app, self.namespace)

    def test_add_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_add_router_required_options(self):
        arglist = [self._agent.id, self._router.id, '--l3']
        verifylist = [('l3', True), ('agent_id', self._agent.id), ('router', self._router.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.add_router_to_agent.assert_called_with(self._agent, self._router)
        self.assertIsNone(result)