from unittest import mock
from openstackclient.identity.v3 import endpoint_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestRemoveProjectEndpointGroup(TestEndpointGroup):
    project = identity_fakes.FakeProject.create_one_project()
    domain = identity_fakes.FakeDomain.create_one_domain()
    endpoint_group = identity_fakes.FakeEndpointGroup.create_one_endpointgroup()

    def setUp(self):
        super(TestRemoveProjectEndpointGroup, self).setUp()
        self.endpoint_groups_mock.get.return_value = self.endpoint_group
        self.projects_mock.get.return_value = self.project
        self.domains_mock.get.return_value = self.domain
        self.epf_mock.delete.return_value = None
        self.cmd = endpoint_group.RemoveProjectFromEndpointGroup(self.app, None)

    def test_remove_project_endpoint_no_options(self):
        arglist = [self.endpoint_group.id, self.project.id]
        verifylist = [('endpointgroup', self.endpoint_group.id), ('project', self.project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.epf_mock.delete_endpoint_group_from_project.assert_called_with(project=self.project.id, endpoint_group=self.endpoint_group.id)
        self.assertIsNone(result)

    def test_remove_project_endpoint_with_options(self):
        arglist = [self.endpoint_group.id, self.project.id, '--project-domain', self.domain.id]
        verifylist = [('endpointgroup', self.endpoint_group.id), ('project', self.project.id), ('project_domain', self.domain.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.epf_mock.delete_endpoint_group_from_project.assert_called_with(project=self.project.id, endpoint_group=self.endpoint_group.id)
        self.assertIsNone(result)