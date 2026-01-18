from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_scope
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestCreateAddressScope(TestAddressScope):
    project = identity_fakes_v3.FakeProject.create_one_project()
    domain = identity_fakes_v3.FakeDomain.create_one_domain()
    new_address_scope = network_fakes.create_one_address_scope(attrs={'project_id': project.id})
    columns = ('id', 'ip_version', 'name', 'project_id', 'shared')
    data = (new_address_scope.id, new_address_scope.ip_version, new_address_scope.name, new_address_scope.project_id, new_address_scope.is_shared)

    def setUp(self):
        super(TestCreateAddressScope, self).setUp()
        self.network_client.create_address_scope = mock.Mock(return_value=self.new_address_scope)
        self.cmd = address_scope.CreateAddressScope(self.app, self.namespace)
        self.projects_mock.get.return_value = self.project
        self.domains_mock.get.return_value = self.domain

    def test_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_default_options(self):
        arglist = [self.new_address_scope.name]
        verifylist = [('project', None), ('ip_version', self.new_address_scope.ip_version), ('name', self.new_address_scope.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_address_scope.assert_called_once_with(**{'ip_version': self.new_address_scope.ip_version, 'name': self.new_address_scope.name})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_all_options(self):
        arglist = ['--ip-version', str(self.new_address_scope.ip_version), '--share', '--project', self.project.name, '--project-domain', self.domain.name, self.new_address_scope.name]
        verifylist = [('ip_version', self.new_address_scope.ip_version), ('share', True), ('project', self.project.name), ('project_domain', self.domain.name), ('name', self.new_address_scope.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_address_scope.assert_called_once_with(**{'ip_version': self.new_address_scope.ip_version, 'shared': True, 'project_id': self.project.id, 'name': self.new_address_scope.name})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_no_share(self):
        arglist = ['--no-share', self.new_address_scope.name]
        verifylist = [('no_share', True), ('name', self.new_address_scope.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_address_scope.assert_called_once_with(**{'ip_version': self.new_address_scope.ip_version, 'shared': False, 'name': self.new_address_scope.name})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)