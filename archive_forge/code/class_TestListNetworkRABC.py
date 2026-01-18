from unittest import mock
from unittest.mock import call
import ddt
from osc_lib import exceptions
from openstackclient.network.v2 import network_rbac
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListNetworkRABC(TestNetworkRBAC):
    rbac_policies = network_fakes.create_network_rbacs(count=3)
    columns = ('ID', 'Object Type', 'Object ID')
    columns_long = ('ID', 'Object Type', 'Object ID', 'Action')
    data = []
    for r in rbac_policies:
        data.append((r.id, r.object_type, r.object_id))
    data_long = []
    for r in rbac_policies:
        data_long.append((r.id, r.object_type, r.object_id, r.action))

    def setUp(self):
        super(TestListNetworkRABC, self).setUp()
        self.cmd = network_rbac.ListNetworkRBAC(self.app, self.namespace)
        self.network_client.rbac_policies = mock.Mock(return_value=self.rbac_policies)
        self.project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = self.project

    def test_network_rbac_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.rbac_policies.assert_called_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_network_rbac_list_type_opt(self):
        arglist = ['--type', self.rbac_policies[0].object_type]
        verifylist = [('type', self.rbac_policies[0].object_type)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.rbac_policies.assert_called_with(**{'object_type': self.rbac_policies[0].object_type})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_network_rbac_list_action_opt(self):
        arglist = ['--action', self.rbac_policies[0].action]
        verifylist = [('action', self.rbac_policies[0].action)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.rbac_policies.assert_called_with(**{'action': self.rbac_policies[0].action})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_network_rbac_list_with_long(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.rbac_policies.assert_called_with()
        self.assertEqual(self.columns_long, columns)
        self.assertEqual(self.data_long, list(data))

    def test_network_rbac_list_target_project_opt(self):
        arglist = ['--target-project', self.rbac_policies[0].target_project_id]
        verifylist = [('target_project', self.rbac_policies[0].target_project_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.rbac_policies.assert_called_with(**{'target_project_id': self.project.id})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))