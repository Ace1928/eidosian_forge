from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_policy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestCreateNetworkQosPolicy(TestQosPolicy):
    project = identity_fakes_v3.FakeProject.create_one_project()
    new_qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy(attrs={'project_id': project.id})
    columns = ('description', 'id', 'is_default', 'name', 'project_id', 'rules', 'shared')
    data = (new_qos_policy.description, new_qos_policy.id, new_qos_policy.is_default, new_qos_policy.name, new_qos_policy.project_id, new_qos_policy.rules, new_qos_policy.shared)

    def setUp(self):
        super(TestCreateNetworkQosPolicy, self).setUp()
        self.network_client.create_qos_policy = mock.Mock(return_value=self.new_qos_policy)
        self.cmd = network_qos_policy.CreateNetworkQosPolicy(self.app, self.namespace)
        self.projects_mock.get.return_value = self.project

    def test_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_default_options(self):
        arglist = [self.new_qos_policy.name]
        verifylist = [('project', None), ('name', self.new_qos_policy.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_qos_policy.assert_called_once_with(**{'name': self.new_qos_policy.name})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_all_options(self):
        arglist = ['--share', '--project', self.project.name, self.new_qos_policy.name, '--description', 'QoS policy description', '--default']
        verifylist = [('share', True), ('project', self.project.name), ('name', self.new_qos_policy.name), ('description', 'QoS policy description'), ('default', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_qos_policy.assert_called_once_with(**{'shared': True, 'project_id': self.project.id, 'name': self.new_qos_policy.name, 'description': 'QoS policy description', 'is_default': True})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_no_default(self):
        arglist = [self.new_qos_policy.name, '--no-default']
        verifylist = [('project', None), ('name', self.new_qos_policy.name), ('default', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_qos_policy.assert_called_once_with(**{'name': self.new_qos_policy.name, 'is_default': False})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)