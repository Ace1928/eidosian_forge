from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import project
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestProjectSet(TestProject):

    def setUp(self):
        super(TestProjectSet, self).setUp()
        self.projects_mock.get.return_value = self.fake_project
        self.projects_mock.update.return_value = self.fake_project
        self.cmd = project.SetProject(self.app, None)

    def test_project_set_no_options(self):
        arglist = [self.fake_project.name]
        verifylist = [('project', self.fake_project.name), ('enable', False), ('disable', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)

    def test_project_set_unexist_project(self):
        arglist = ['unexist-project']
        verifylist = [('project', 'unexist-project'), ('name', None), ('description', None), ('enable', False), ('disable', False), ('property', None)]
        self.projects_mock.get.side_effect = exceptions.NotFound(None)
        self.projects_mock.find.side_effect = exceptions.NotFound(None)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_project_set_name(self):
        arglist = ['--name', self.fake_project.name, self.fake_project.name]
        verifylist = [('name', self.fake_project.name), ('enable', False), ('disable', False), ('project', self.fake_project.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'description': self.fake_project.description, 'enabled': True, 'tenant_name': self.fake_project.name}
        self.projects_mock.update.assert_called_with(self.fake_project.id, **kwargs)
        self.assertIsNone(result)

    def test_project_set_description(self):
        arglist = ['--description', self.fake_project.description, self.fake_project.name]
        verifylist = [('description', self.fake_project.description), ('enable', False), ('disable', False), ('project', self.fake_project.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'description': self.fake_project.description, 'enabled': True, 'tenant_name': self.fake_project.name}
        self.projects_mock.update.assert_called_with(self.fake_project.id, **kwargs)
        self.assertIsNone(result)

    def test_project_set_enable(self):
        arglist = ['--enable', self.fake_project.name]
        verifylist = [('enable', True), ('disable', False), ('project', self.fake_project.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'description': self.fake_project.description, 'enabled': True, 'tenant_name': self.fake_project.name}
        self.projects_mock.update.assert_called_with(self.fake_project.id, **kwargs)
        self.assertIsNone(result)

    def test_project_set_disable(self):
        arglist = ['--disable', self.fake_project.name]
        verifylist = [('enable', False), ('disable', True), ('project', self.fake_project.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'description': self.fake_project.description, 'enabled': False, 'tenant_name': self.fake_project.name}
        self.projects_mock.update.assert_called_with(self.fake_project.id, **kwargs)
        self.assertIsNone(result)

    def test_project_set_property(self):
        arglist = ['--property', 'fee=fi', '--property', 'fo=fum', self.fake_project.name]
        verifylist = [('property', {'fee': 'fi', 'fo': 'fum'}), ('project', self.fake_project.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'description': self.fake_project.description, 'enabled': True, 'tenant_name': self.fake_project.name, 'fee': 'fi', 'fo': 'fum'}
        self.projects_mock.update.assert_called_with(self.fake_project.id, **kwargs)
        self.assertIsNone(result)