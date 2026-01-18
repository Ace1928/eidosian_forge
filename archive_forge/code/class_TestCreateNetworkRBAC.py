from unittest import mock
from unittest.mock import call
import ddt
from osc_lib import exceptions
from openstackclient.network.v2 import network_rbac
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
@ddt.ddt
class TestCreateNetworkRBAC(TestNetworkRBAC):
    network_object = network_fakes.create_one_network()
    qos_object = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
    sg_object = network_fakes.FakeNetworkSecGroup.create_one_security_group()
    as_object = network_fakes.create_one_address_scope()
    snp_object = network_fakes.FakeSubnetPool.create_one_subnet_pool()
    ag_object = network_fakes.create_one_address_group()
    project = identity_fakes_v3.FakeProject.create_one_project()
    rbac_policy = network_fakes.create_one_network_rbac(attrs={'project_id': project.id, 'target_tenant': project.id, 'object_id': network_object.id})
    columns = ('action', 'id', 'object_id', 'object_type', 'project_id', 'target_project_id')
    data = [rbac_policy.action, rbac_policy.id, rbac_policy.object_id, rbac_policy.object_type, rbac_policy.project_id, rbac_policy.target_project_id]

    def setUp(self):
        super(TestCreateNetworkRBAC, self).setUp()
        self.cmd = network_rbac.CreateNetworkRBAC(self.app, self.namespace)
        self.network_client.create_rbac_policy = mock.Mock(return_value=self.rbac_policy)
        self.network_client.find_network = mock.Mock(return_value=self.network_object)
        self.network_client.find_qos_policy = mock.Mock(return_value=self.qos_object)
        self.network_client.find_security_group = mock.Mock(return_value=self.sg_object)
        self.network_client.find_address_scope = mock.Mock(return_value=self.as_object)
        self.network_client.find_subnet_pool = mock.Mock(return_value=self.snp_object)
        self.network_client.find_address_group = mock.Mock(return_value=self.ag_object)
        self.projects_mock.get.return_value = self.project

    def test_network_rbac_create_no_type(self):
        arglist = ['--action', self.rbac_policy.action, self.rbac_policy.object_id]
        verifylist = [('action', self.rbac_policy.action), ('rbac_policy', self.rbac_policy.id)]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_network_rbac_create_no_action(self):
        arglist = ['--type', self.rbac_policy.object_type, self.rbac_policy.object_id]
        verifylist = [('type', self.rbac_policy.object_type), ('rbac_policy', self.rbac_policy.id)]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_network_rbac_create_invalid_type(self):
        arglist = ['--action', self.rbac_policy.action, '--type', 'invalid_type', '--target-project', self.rbac_policy.target_project_id, self.rbac_policy.object_id]
        verifylist = [('action', self.rbac_policy.action), ('type', 'invalid_type'), ('target-project', self.rbac_policy.target_project_id), ('rbac_policy', self.rbac_policy.id)]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_network_rbac_create_invalid_action(self):
        arglist = ['--type', self.rbac_policy.object_type, '--action', 'invalid_action', '--target-project', self.rbac_policy.target_project_id, self.rbac_policy.object_id]
        verifylist = [('type', self.rbac_policy.object_type), ('action', 'invalid_action'), ('target-project', self.rbac_policy.target_project_id), ('rbac_policy', self.rbac_policy.id)]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_network_rbac_create(self):
        arglist = ['--type', self.rbac_policy.object_type, '--action', self.rbac_policy.action, '--target-project', self.rbac_policy.target_project_id, self.rbac_policy.object_id]
        verifylist = [('type', self.rbac_policy.object_type), ('action', self.rbac_policy.action), ('target_project', self.rbac_policy.target_project_id), ('rbac_object', self.rbac_policy.object_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_rbac_policy.assert_called_with(**{'object_id': self.rbac_policy.object_id, 'object_type': self.rbac_policy.object_type, 'action': self.rbac_policy.action, 'target_tenant': self.rbac_policy.target_project_id})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_network_rbac_create_with_target_all_projects(self):
        arglist = ['--type', self.rbac_policy.object_type, '--action', self.rbac_policy.action, '--target-all-projects', self.rbac_policy.object_id]
        verifylist = [('type', self.rbac_policy.object_type), ('action', self.rbac_policy.action), ('target_all_projects', True), ('rbac_object', self.rbac_policy.object_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_rbac_policy.assert_called_with(**{'object_id': self.rbac_policy.object_id, 'object_type': self.rbac_policy.object_type, 'action': self.rbac_policy.action, 'target_tenant': '*'})

    def test_network_rbac_create_all_options(self):
        arglist = ['--type', self.rbac_policy.object_type, '--action', self.rbac_policy.action, '--target-project', self.rbac_policy.target_project_id, '--project', self.rbac_policy.project_id, '--project-domain', self.project.domain_id, '--target-project-domain', self.project.domain_id, self.rbac_policy.object_id]
        verifylist = [('type', self.rbac_policy.object_type), ('action', self.rbac_policy.action), ('target_project', self.rbac_policy.target_project_id), ('project', self.rbac_policy.project_id), ('project_domain', self.project.domain_id), ('target_project_domain', self.project.domain_id), ('rbac_object', self.rbac_policy.object_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_rbac_policy.assert_called_with(**{'object_id': self.rbac_policy.object_id, 'object_type': self.rbac_policy.object_type, 'action': self.rbac_policy.action, 'target_tenant': self.rbac_policy.target_project_id, 'project_id': self.rbac_policy.project_id})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    @ddt.data(('qos_policy', 'qos_object'), ('security_group', 'sg_object'), ('subnetpool', 'snp_object'), ('address_scope', 'as_object'), ('address_group', 'ag_object'))
    @ddt.unpack
    def test_network_rbac_create_object(self, obj_type, obj_fake_attr):
        obj_fake = getattr(self, obj_fake_attr)
        self.rbac_policy.object_type = obj_type
        self.rbac_policy.object_id = obj_fake.id
        arglist = ['--type', obj_type, '--action', self.rbac_policy.action, '--target-project', self.rbac_policy.target_project_id, obj_fake.name]
        verifylist = [('type', obj_type), ('action', self.rbac_policy.action), ('target_project', self.rbac_policy.target_project_id), ('rbac_object', obj_fake.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_rbac_policy.assert_called_with(**{'object_id': obj_fake.id, 'object_type': obj_type, 'action': self.rbac_policy.action, 'target_tenant': self.rbac_policy.target_project_id})
        self.data = [self.rbac_policy.action, self.rbac_policy.id, obj_fake.id, obj_type, self.rbac_policy.project_id, self.rbac_policy.target_project_id]
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))