from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import security_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListSecurityGroupNetwork(TestSecurityGroupNetwork):
    _security_groups = network_fakes.FakeSecurityGroup.create_security_groups(count=3)
    columns = ('ID', 'Name', 'Description', 'Project', 'Tags')
    data = []
    for grp in _security_groups:
        data.append((grp.id, grp.name, grp.description, grp.project_id, grp.tags))

    def setUp(self):
        super(TestListSecurityGroupNetwork, self).setUp()
        self.network_client.security_groups = mock.Mock(return_value=self._security_groups)
        self.cmd = security_group.ListSecurityGroup(self.app, self.namespace)

    def test_security_group_list_no_options(self):
        arglist = []
        verifylist = [('all_projects', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.security_groups.assert_called_once_with(fields=security_group.ListSecurityGroup.FIELDS_TO_RETRIEVE)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_security_group_list_all_projects(self):
        arglist = ['--all-projects']
        verifylist = [('all_projects', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.security_groups.assert_called_once_with(fields=security_group.ListSecurityGroup.FIELDS_TO_RETRIEVE)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_security_group_list_project(self):
        project = identity_fakes.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id, 'fields': security_group.ListSecurityGroup.FIELDS_TO_RETRIEVE}
        self.network_client.security_groups.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_security_group_list_project_domain(self):
        project = identity_fakes.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id, '--project-domain', project.domain_id]
        verifylist = [('project', project.id), ('project_domain', project.domain_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id, 'fields': security_group.ListSecurityGroup.FIELDS_TO_RETRIEVE}
        self.network_client.security_groups.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_list_with_tag_options(self):
        arglist = ['--tags', 'red,blue', '--any-tags', 'red,green', '--not-tags', 'orange,yellow', '--not-any-tags', 'black,white']
        verifylist = [('tags', ['red', 'blue']), ('any_tags', ['red', 'green']), ('not_tags', ['orange', 'yellow']), ('not_any_tags', ['black', 'white'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.security_groups.assert_called_once_with(**{'tags': 'red,blue', 'any_tags': 'red,green', 'not_tags': 'orange,yellow', 'not_any_tags': 'black,white', 'fields': security_group.ListSecurityGroup.FIELDS_TO_RETRIEVE})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))