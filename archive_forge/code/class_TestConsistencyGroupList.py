from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
class TestConsistencyGroupList(TestConsistencyGroup):
    consistency_groups = volume_fakes.create_consistency_groups(count=2)
    columns = ['ID', 'Status', 'Name']
    columns_long = ['ID', 'Status', 'Availability Zone', 'Name', 'Description', 'Volume Types']
    data = []
    for c in consistency_groups:
        data.append((c.id, c.status, c.name))
    data_long = []
    for c in consistency_groups:
        data_long.append((c.id, c.status, c.availability_zone, c.name, c.description, format_columns.ListColumn(c.volume_types)))

    def setUp(self):
        super().setUp()
        self.consistencygroups_mock.list.return_value = self.consistency_groups
        self.cmd = consistency_group.ListConsistencyGroup(self.app, None)

    def test_consistency_group_list_without_options(self):
        arglist = []
        verifylist = [('all_projects', False), ('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.consistencygroups_mock.list.assert_called_once_with(detailed=True, search_opts={'all_tenants': False})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_consistency_group_list_with_all_project(self):
        arglist = ['--all-projects']
        verifylist = [('all_projects', True), ('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.consistencygroups_mock.list.assert_called_once_with(detailed=True, search_opts={'all_tenants': True})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_consistency_group_list_with_long(self):
        arglist = ['--long']
        verifylist = [('all_projects', False), ('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.consistencygroups_mock.list.assert_called_once_with(detailed=True, search_opts={'all_tenants': False})
        self.assertEqual(self.columns_long, columns)
        self.assertCountEqual(self.data_long, list(data))