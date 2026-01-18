from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_scope
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListAddressScope(TestAddressScope):
    address_scopes = network_fakes.create_address_scopes(count=3)
    columns = ('ID', 'Name', 'IP Version', 'Shared', 'Project')
    data = []
    for scope in address_scopes:
        data.append((scope.id, scope.name, scope.ip_version, scope.is_shared, scope.project_id))

    def setUp(self):
        super(TestListAddressScope, self).setUp()
        self.network_client.address_scopes = mock.Mock(return_value=self.address_scopes)
        self.cmd = address_scope.ListAddressScope(self.app, self.namespace)

    def test_address_scope_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.address_scopes.assert_called_once_with(**{})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_address_scope_list_name(self):
        arglist = ['--name', self.address_scopes[0].name]
        verifylist = [('name', self.address_scopes[0].name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.address_scopes.assert_called_once_with(**{'name': self.address_scopes[0].name})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_address_scope_list_ip_version(self):
        arglist = ['--ip-version', str(4)]
        verifylist = [('ip_version', 4)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.address_scopes.assert_called_once_with(**{'ip_version': 4})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_address_scope_list_project(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.address_scopes.assert_called_once_with(**{'project_id': project.id})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_address_scope_project_domain(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id, '--project-domain', project.domain_id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id}
        self.network_client.address_scopes.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_address_scope_list_share(self):
        arglist = ['--share']
        verifylist = [('share', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.address_scopes.assert_called_once_with(**{'is_shared': True})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_address_scope_list_no_share(self):
        arglist = ['--no-share']
        verifylist = [('no_share', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.address_scopes.assert_called_once_with(**{'is_shared': False})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))