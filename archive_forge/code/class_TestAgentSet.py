from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.compute.v2 import agent
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestAgentSet(TestAgent):

    def setUp(self):
        super(TestAgentSet, self).setUp()
        self.agents_mock.update.return_value = self.fake_agent
        self.agents_mock.list.return_value = [self.fake_agent]
        self.cmd = agent.SetAgent(self.app, None)

    def test_agent_set_nothing(self):
        arglist = ['1']
        verifylist = [('id', '1')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.agents_mock.update.assert_called_with(parsed_args.id, self.fake_agent.version, self.fake_agent.url, self.fake_agent.md5hash)
        self.assertIsNone(result)

    def test_agent_set_version(self):
        arglist = ['1', '--agent-version', 'new-version']
        verifylist = [('id', '1'), ('version', 'new-version')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.agents_mock.update.assert_called_with(parsed_args.id, parsed_args.version, self.fake_agent.url, self.fake_agent.md5hash)
        self.assertIsNone(result)

    def test_agent_set_url(self):
        arglist = ['1', '--url', 'new-url']
        verifylist = [('id', '1'), ('url', 'new-url')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.agents_mock.update.assert_called_with(parsed_args.id, self.fake_agent.version, parsed_args.url, self.fake_agent.md5hash)
        self.assertIsNone(result)

    def test_agent_set_md5hash(self):
        arglist = ['1', '--md5hash', 'new-md5hash']
        verifylist = [('id', '1'), ('md5hash', 'new-md5hash')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.agents_mock.update.assert_called_with(parsed_args.id, self.fake_agent.version, self.fake_agent.url, parsed_args.md5hash)
        self.assertIsNone(result)