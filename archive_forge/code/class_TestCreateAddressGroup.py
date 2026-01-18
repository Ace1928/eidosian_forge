from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestCreateAddressGroup(TestAddressGroup):
    project = identity_fakes_v3.FakeProject.create_one_project()
    domain = identity_fakes_v3.FakeDomain.create_one_domain()
    new_address_group = network_fakes.create_one_address_group(attrs={'project_id': project.id})
    columns = ('addresses', 'description', 'id', 'name', 'project_id')
    data = (new_address_group.addresses, new_address_group.description, new_address_group.id, new_address_group.name, new_address_group.project_id)

    def setUp(self):
        super(TestCreateAddressGroup, self).setUp()
        self.network_client.create_address_group = mock.Mock(return_value=self.new_address_group)
        self.cmd = address_group.CreateAddressGroup(self.app, self.namespace)
        self.projects_mock.get.return_value = self.project
        self.domains_mock.get.return_value = self.domain

    def test_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_default_options(self):
        arglist = [self.new_address_group.name]
        verifylist = [('project', None), ('name', self.new_address_group.name), ('description', None), ('address', [])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_address_group.assert_called_once_with(**{'name': self.new_address_group.name, 'addresses': []})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_all_options(self):
        arglist = ['--project', self.project.name, '--project-domain', self.domain.name, '--address', '10.0.0.1', '--description', self.new_address_group.description, self.new_address_group.name]
        verifylist = [('project', self.project.name), ('project_domain', self.domain.name), ('address', ['10.0.0.1']), ('description', self.new_address_group.description), ('name', self.new_address_group.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_address_group.assert_called_once_with(**{'addresses': ['10.0.0.1/32'], 'project_id': self.project.id, 'name': self.new_address_group.name, 'description': self.new_address_group.description})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)