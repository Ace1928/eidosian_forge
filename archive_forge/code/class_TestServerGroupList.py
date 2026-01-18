from unittest import mock
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import server_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestServerGroupList(TestServerGroup):
    list_columns = ('ID', 'Name', 'Policies')
    list_columns_long = ('ID', 'Name', 'Policies', 'Members', 'Project Id', 'User Id')
    list_columns_v264 = ('ID', 'Name', 'Policy')
    list_columns_v264_long = ('ID', 'Name', 'Policy', 'Members', 'Project Id', 'User Id')
    list_data = ((TestServerGroup.fake_server_group.id, TestServerGroup.fake_server_group.name, format_columns.ListColumn(TestServerGroup.fake_server_group.policies)),)
    list_data_long = ((TestServerGroup.fake_server_group.id, TestServerGroup.fake_server_group.name, format_columns.ListColumn(TestServerGroup.fake_server_group.policies), format_columns.ListColumn(TestServerGroup.fake_server_group.member_ids), TestServerGroup.fake_server_group.project_id, TestServerGroup.fake_server_group.user_id),)
    list_data_v264 = ((TestServerGroup.fake_server_group.id, TestServerGroup.fake_server_group.name, TestServerGroup.fake_server_group.policy),)
    list_data_v264_long = ((TestServerGroup.fake_server_group.id, TestServerGroup.fake_server_group.name, TestServerGroup.fake_server_group.policy, format_columns.ListColumn(TestServerGroup.fake_server_group.member_ids), TestServerGroup.fake_server_group.project_id, TestServerGroup.fake_server_group.user_id),)

    def setUp(self):
        super().setUp()
        self.compute_sdk_client.server_groups.return_value = [self.fake_server_group]
        self.cmd = server_group.ListServerGroup(self.app, None)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_server_group_list(self, sm_mock):
        arglist = []
        verifylist = [('all_projects', False), ('long', False), ('limit', None), ('offset', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.server_groups.assert_called_once_with()
        self.assertCountEqual(self.list_columns, columns)
        self.assertCountEqual(self.list_data, tuple(data))

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_server_group_list_with_all_projects_and_long(self, sm_mock):
        arglist = ['--all-projects', '--long']
        verifylist = [('all_projects', True), ('long', True), ('limit', None), ('offset', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.server_groups.assert_called_once_with(all_projects=True)
        self.assertCountEqual(self.list_columns_long, columns)
        self.assertCountEqual(self.list_data_long, tuple(data))

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_server_group_list_with_limit(self, sm_mock):
        arglist = ['--limit', '1']
        verifylist = [('all_projects', False), ('long', False), ('limit', 1), ('offset', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.server_groups.assert_called_once_with(limit=1)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_server_group_list_with_offset(self, sm_mock):
        arglist = ['--offset', '5']
        verifylist = [('all_projects', False), ('long', False), ('limit', None), ('offset', 5)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.server_groups.assert_called_once_with(offset=5)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_server_group_list_v264(self, sm_mock):
        arglist = []
        verifylist = [('all_projects', False), ('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.server_groups.assert_called_once_with()
        self.assertCountEqual(self.list_columns_v264, columns)
        self.assertCountEqual(self.list_data_v264, tuple(data))

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_server_group_list_with_all_projects_and_long_v264(self, sm_mock):
        arglist = ['--all-projects', '--long']
        verifylist = [('all_projects', True), ('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.server_groups.assert_called_once_with(all_projects=True)
        self.assertCountEqual(self.list_columns_v264_long, columns)
        self.assertCountEqual(self.list_data_v264_long, tuple(data))