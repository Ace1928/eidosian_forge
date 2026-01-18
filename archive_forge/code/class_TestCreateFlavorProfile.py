from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor_profile
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
class TestCreateFlavorProfile(TestFlavorProfile):
    project = identity_fakes_v3.FakeProject.create_one_project()
    domain = identity_fakes_v3.FakeDomain.create_one_domain()
    new_flavor_profile = network_fakes.create_one_service_profile()
    columns = ('description', 'driver', 'enabled', 'id', 'meta_info', 'project_id')
    data = (new_flavor_profile.description, new_flavor_profile.driver, new_flavor_profile.is_enabled, new_flavor_profile.id, new_flavor_profile.meta_info, new_flavor_profile.project_id)

    def setUp(self):
        super(TestCreateFlavorProfile, self).setUp()
        self.network_client.create_service_profile = mock.Mock(return_value=self.new_flavor_profile)
        self.projects_mock.get.return_value = self.project
        self.cmd = network_flavor_profile.CreateNetworkFlavorProfile(self.app, self.namespace)

    def test_create_all_options(self):
        arglist = ['--description', self.new_flavor_profile.description, '--project', self.new_flavor_profile.project_id, '--project-domain', self.domain.name, '--enable', '--driver', self.new_flavor_profile.driver, '--metainfo', self.new_flavor_profile.meta_info]
        verifylist = [('description', self.new_flavor_profile.description), ('project', self.new_flavor_profile.project_id), ('project_domain', self.domain.name), ('enable', True), ('driver', self.new_flavor_profile.driver), ('metainfo', self.new_flavor_profile.meta_info)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_service_profile.assert_called_once_with(**{'description': self.new_flavor_profile.description, 'project_id': self.project.id, 'enabled': self.new_flavor_profile.is_enabled, 'driver': self.new_flavor_profile.driver, 'metainfo': self.new_flavor_profile.meta_info})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_with_metainfo(self):
        arglist = ['--description', self.new_flavor_profile.description, '--project', self.new_flavor_profile.project_id, '--project-domain', self.domain.name, '--enable', '--metainfo', self.new_flavor_profile.meta_info]
        verifylist = [('description', self.new_flavor_profile.description), ('project', self.new_flavor_profile.project_id), ('project_domain', self.domain.name), ('enable', True), ('metainfo', self.new_flavor_profile.meta_info)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_service_profile.assert_called_once_with(**{'description': self.new_flavor_profile.description, 'project_id': self.project.id, 'enabled': self.new_flavor_profile.is_enabled, 'metainfo': self.new_flavor_profile.meta_info})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_with_driver(self):
        arglist = ['--description', self.new_flavor_profile.description, '--project', self.new_flavor_profile.project_id, '--project-domain', self.domain.name, '--enable', '--driver', self.new_flavor_profile.driver]
        verifylist = [('description', self.new_flavor_profile.description), ('project', self.new_flavor_profile.project_id), ('project_domain', self.domain.name), ('enable', True), ('driver', self.new_flavor_profile.driver)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_service_profile.assert_called_once_with(**{'description': self.new_flavor_profile.description, 'project_id': self.project.id, 'enabled': self.new_flavor_profile.is_enabled, 'driver': self.new_flavor_profile.driver})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_without_driver_and_metainfo(self):
        arglist = ['--description', self.new_flavor_profile.description, '--project', self.new_flavor_profile.project_id, '--project-domain', self.domain.name, '--enable']
        verifylist = [('description', self.new_flavor_profile.description), ('project', self.new_flavor_profile.project_id), ('project_domain', self.domain.name), ('enable', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_create_disable(self):
        arglist = ['--disable', '--driver', self.new_flavor_profile.driver]
        verifylist = [('disable', True), ('driver', self.new_flavor_profile.driver)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_service_profile.assert_called_once_with(**{'enabled': False, 'driver': self.new_flavor_profile.driver})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)