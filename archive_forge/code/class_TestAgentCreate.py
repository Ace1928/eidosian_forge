from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.compute.v2 import agent
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestAgentCreate(TestAgent):

    def setUp(self):
        super(TestAgentCreate, self).setUp()
        self.agents_mock.create.return_value = self.fake_agent
        self.cmd = agent.CreateAgent(self.app, None)

    def test_agent_create(self):
        arglist = [self.fake_agent.os, self.fake_agent.architecture, self.fake_agent.version, self.fake_agent.url, self.fake_agent.md5hash, self.fake_agent.hypervisor]
        verifylist = [('os', self.fake_agent.os), ('architecture', self.fake_agent.architecture), ('version', self.fake_agent.version), ('url', self.fake_agent.url), ('md5hash', self.fake_agent.md5hash), ('hypervisor', self.fake_agent.hypervisor)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.agents_mock.create.assert_called_with(parsed_args.os, parsed_args.architecture, parsed_args.version, parsed_args.url, parsed_args.md5hash, parsed_args.hypervisor)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)