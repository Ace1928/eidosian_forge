from unittest import mock
from openstackclient.network.v2 import network_auto_allocated_topology
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
class TestValidateAutoAllocatedTopology(TestAutoAllocatedTopology):
    project = identity_fakes.FakeProject.create_one_project()
    network_object = network_fakes.create_one_network()
    topology = network_fakes.create_one_topology(attrs={'id': network_object.id, 'project_id': project.id})
    columns = ('id', 'project_id')
    data = (network_object.id, project.id)

    def setUp(self):
        super(TestValidateAutoAllocatedTopology, self).setUp()
        self.cmd = network_auto_allocated_topology.CreateAutoAllocatedTopology(self.app, self.namespace)
        self.network_client.validate_auto_allocated_topology = mock.Mock(return_value=self.topology)

    def test_show_dry_run_no_project(self):
        arglist = ['--check-resources']
        verifylist = [('check_resources', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.validate_auto_allocated_topology.assert_called_with(None)

    def test_show_dry_run_project_option(self):
        arglist = ['--check-resources', '--project', self.project.id]
        verifylist = [('check_resources', True), ('project', self.project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.validate_auto_allocated_topology.assert_called_with(self.project.id)

    def test_show_dry_run_project_domain_option(self):
        arglist = ['--check-resources', '--project', self.project.id, '--project-domain', self.project.domain_id]
        verifylist = [('check_resources', True), ('project', self.project.id), ('project_domain', self.project.domain_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.validate_auto_allocated_topology.assert_called_with(self.project.id)