from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import project
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestProjectCreate(TestProject):

    def setUp(self):
        super(TestProjectCreate, self).setUp()
        self.projects_mock.create.return_value = self.fake_project
        self.cmd = project.CreateProject(self.app, None)

    def test_project_create_no_options(self):
        arglist = [self.fake_project.name]
        verifylist = [('enable', False), ('disable', False), ('name', self.fake_project.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'description': None, 'enabled': True}
        self.projects_mock.create.assert_called_with(self.fake_project.name, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_project_create_description(self):
        arglist = ['--description', 'new desc', self.fake_project.name]
        verifylist = [('description', 'new desc'), ('name', self.fake_project.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'description': 'new desc', 'enabled': True}
        self.projects_mock.create.assert_called_with(self.fake_project.name, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_project_create_enable(self):
        arglist = ['--enable', self.fake_project.name]
        verifylist = [('enable', True), ('disable', False), ('name', self.fake_project.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'description': None, 'enabled': True}
        self.projects_mock.create.assert_called_with(self.fake_project.name, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_project_create_disable(self):
        arglist = ['--disable', self.fake_project.name]
        verifylist = [('enable', False), ('disable', True), ('name', self.fake_project.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'description': None, 'enabled': False}
        self.projects_mock.create.assert_called_with(self.fake_project.name, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_project_create_property(self):
        arglist = ['--property', 'fee=fi', '--property', 'fo=fum', self.fake_project.name]
        verifylist = [('property', {'fee': 'fi', 'fo': 'fum'}), ('name', self.fake_project.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'description': None, 'enabled': True, 'fee': 'fi', 'fo': 'fum'}
        self.projects_mock.create.assert_called_with(self.fake_project.name, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_project_create_or_show_exists(self):

        def _raise_conflict(*args, **kwargs):
            raise ks_exc.Conflict(None)
        self.projects_mock.create.side_effect = _raise_conflict
        self.projects_mock.get.return_value = self.fake_project
        arglist = ['--or-show', self.fake_project.name]
        verifylist = [('or_show', True), ('name', self.fake_project.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.projects_mock.get.assert_called_with(self.fake_project.name)
        kwargs = {'description': None, 'enabled': True}
        self.projects_mock.create.assert_called_with(self.fake_project.name, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_project_create_or_show_not_exists(self):
        arglist = ['--or-show', self.fake_project.name]
        verifylist = [('or_show', True), ('name', self.fake_project.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'description': None, 'enabled': True}
        self.projects_mock.create.assert_called_with(self.fake_project.name, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)