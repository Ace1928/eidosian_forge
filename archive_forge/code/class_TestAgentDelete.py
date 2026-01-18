from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.compute.v2 import agent
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestAgentDelete(TestAgent):
    fake_agents = compute_fakes.create_agents(count=2)

    def setUp(self):
        super(TestAgentDelete, self).setUp()
        self.agents_mock.get.return_value = self.fake_agents
        self.cmd = agent.DeleteAgent(self.app, None)

    def test_delete_one_agent(self):
        arglist = [self.fake_agents[0].agent_id]
        verifylist = [('id', [self.fake_agents[0].agent_id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.agents_mock.delete.assert_called_with(self.fake_agents[0].agent_id)
        self.assertIsNone(result)

    def test_delete_multiple_agents(self):
        arglist = []
        for n in self.fake_agents:
            arglist.append(n.agent_id)
        verifylist = [('id', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for n in self.fake_agents:
            calls.append(call(n.agent_id))
        self.agents_mock.delete.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_delete_multiple_agents_exception(self):
        arglist = [self.fake_agents[0].agent_id, self.fake_agents[1].agent_id, 'x-y-z']
        verifylist = [('id', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ret_delete = [None, None, exceptions.NotFound('404')]
        self.agents_mock.delete = mock.Mock(side_effect=ret_delete)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        calls = [call(self.fake_agents[0].agent_id), call(self.fake_agents[1].agent_id)]
        self.agents_mock.delete.assert_has_calls(calls)

    def test_agent_delete_no_input(self):
        arglist = []
        verifylist = None
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)