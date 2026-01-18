from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestCreateNetworkFlavor(TestNetworkFlavor):
    project = identity_fakes_v3.FakeProject.create_one_project()
    domain = identity_fakes_v3.FakeDomain.create_one_domain()
    new_network_flavor = network_fakes.create_one_network_flavor()
    columns = ('description', 'enabled', 'id', 'name', 'service_type', 'service_profile_ids')
    data = (new_network_flavor.description, new_network_flavor.is_enabled, new_network_flavor.id, new_network_flavor.name, new_network_flavor.service_type, new_network_flavor.service_profile_ids)

    def setUp(self):
        super(TestCreateNetworkFlavor, self).setUp()
        self.network_client.create_flavor = mock.Mock(return_value=self.new_network_flavor)
        self.cmd = network_flavor.CreateNetworkFlavor(self.app, self.namespace)
        self.projects_mock.get.return_value = self.project
        self.domains_mock.get.return_value = self.domain

    def test_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_default_options(self):
        arglist = ['--service-type', self.new_network_flavor.service_type, self.new_network_flavor.name]
        verifylist = [('service_type', self.new_network_flavor.service_type), ('name', self.new_network_flavor.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_flavor.assert_called_once_with(**{'service_type': self.new_network_flavor.service_type, 'name': self.new_network_flavor.name})
        self.assertEqual(set(self.columns), set(columns))
        self.assertEqual(set(self.data), set(data))

    def test_create_all_options(self):
        arglist = ['--description', self.new_network_flavor.description, '--enable', '--project', self.project.id, '--project-domain', self.domain.name, '--service-type', self.new_network_flavor.service_type, self.new_network_flavor.name]
        verifylist = [('description', self.new_network_flavor.description), ('enable', True), ('project', self.project.id), ('project_domain', self.domain.name), ('service_type', self.new_network_flavor.service_type), ('name', self.new_network_flavor.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_flavor.assert_called_once_with(**{'description': self.new_network_flavor.description, 'enabled': True, 'project_id': self.project.id, 'service_type': self.new_network_flavor.service_type, 'name': self.new_network_flavor.name})
        self.assertEqual(set(self.columns), set(columns))
        self.assertEqual(set(self.data), set(data))

    def test_create_disable(self):
        arglist = ['--disable', '--service-type', self.new_network_flavor.service_type, self.new_network_flavor.name]
        verifylist = [('disable', True), ('service_type', self.new_network_flavor.service_type), ('name', self.new_network_flavor.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_flavor.assert_called_once_with(**{'enabled': False, 'service_type': self.new_network_flavor.service_type, 'name': self.new_network_flavor.name})
        self.assertEqual(set(self.columns), set(columns))
        self.assertEqual(set(self.data), set(data))