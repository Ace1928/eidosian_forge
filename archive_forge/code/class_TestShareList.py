import argparse
import ddt
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.common.apiclient import exceptions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share as osc_shares
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareList(TestShare):
    project = identity_fakes.FakeProject.create_one_project()
    user = identity_fakes.FakeUser.create_one_user()
    columns = ['ID', 'Name', 'Size', 'Share Proto', 'Status', 'Is Public', 'Share Type Name', 'Host', 'Availability Zone']

    def setUp(self):
        super(TestShareList, self).setUp()
        self.new_share = manila_fakes.FakeShare.create_one_share()
        self.shares_mock.list.return_value = [self.new_share]
        self.users_mock.get.return_value = self.user
        self.projects_mock.get.return_value = self.project
        self.cmd = osc_shares.ListShare(self.app, None)

    def _get_data(self):
        data = ((self.new_share.id, self.new_share.name, self.new_share.size, self.new_share.share_proto, self.new_share.status, self.new_share.is_public, self.new_share.share_type_name, self.new_share.host, self.new_share.availability_zone),)
        return data

    def _get_search_opts(self):
        search_opts = {'all_tenants': False, 'is_public': False, 'metadata': {}, 'extra_specs': {}, 'limit': None, 'name': None, 'status': None, 'host': None, 'share_server_id': None, 'share_network_id': None, 'share_type_id': None, 'snapshot_id': None, 'share_group_id': None, 'project_id': None, 'user_id': None, 'offset': None, 'is_soft_deleted': False, 'export_location': None, 'name~': None, 'description~': None}
        return search_opts

    def test_share_list_no_options(self):
        arglist = []
        verifylist = [('long', False), ('all_projects', False), ('name', None), ('status', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cmd_columns, cmd_data = self.cmd.take_action(parsed_args)
        search_opts = self._get_search_opts()
        self.shares_mock.list.assert_called_once_with(search_opts=search_opts)
        self.assertEqual(self.columns, cmd_columns)
        data = self._get_data()
        self.assertEqual(data, tuple(cmd_data))

    def test_share_list_project(self):
        arglist = ['--project', self.project.name]
        verifylist = [('project', self.project.name), ('long', False), ('all_projects', False), ('status', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cmd_columns, cmd_data = self.cmd.take_action(parsed_args)
        search_opts = self._get_search_opts()
        search_opts['project_id'] = self.project.id
        search_opts['all_tenants'] = True
        self.shares_mock.list.assert_called_once_with(search_opts=search_opts)
        self.assertEqual(self.columns, cmd_columns)
        data = self._get_data()
        self.assertEqual(data, tuple(cmd_data))

    def test_share_list_project_domain(self):
        arglist = ['--project', self.project.name, '--project-domain', self.project.domain_id]
        verifylist = [('project', self.project.name), ('project_domain', self.project.domain_id), ('long', False), ('all_projects', False), ('status', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cmd_columns, cmd_data = self.cmd.take_action(parsed_args)
        search_opts = self._get_search_opts()
        search_opts['project_id'] = self.project.id
        search_opts['all_tenants'] = True
        self.shares_mock.list.assert_called_once_with(search_opts=search_opts)
        self.assertEqual(self.columns, cmd_columns)
        data = self._get_data()
        self.assertEqual(data, tuple(cmd_data))

    def test_share_list_user(self):
        arglist = ['--user', self.user.name]
        verifylist = [('user', self.user.name), ('long', False), ('all_projects', False), ('status', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cmd_columns, cmd_data = self.cmd.take_action(parsed_args)
        search_opts = self._get_search_opts()
        search_opts['user_id'] = self.user.id
        self.shares_mock.list.assert_called_once_with(search_opts=search_opts)
        self.assertEqual(self.columns, cmd_columns)
        data = self._get_data()
        self.assertEqual(data, tuple(cmd_data))

    def test_share_list_user_domain(self):
        arglist = ['--user', self.user.name, '--user-domain', self.user.domain_id]
        verifylist = [('user', self.user.name), ('user_domain', self.user.domain_id), ('long', False), ('all_projects', False), ('status', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cmd_columns, cmd_data = self.cmd.take_action(parsed_args)
        search_opts = self._get_search_opts()
        search_opts['user_id'] = self.user.id
        self.shares_mock.list.assert_called_once_with(search_opts=search_opts)
        self.assertEqual(self.columns, cmd_columns)
        data = self._get_data()
        self.assertEqual(data, tuple(cmd_data))

    def test_share_list_name(self):
        arglist = ['--name', self.new_share.name]
        verifylist = [('long', False), ('all_projects', False), ('name', self.new_share.name), ('status', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cmd_columns, cmd_data = self.cmd.take_action(parsed_args)
        search_opts = self._get_search_opts()
        search_opts['name'] = self.new_share.name
        self.shares_mock.list.assert_called_once_with(search_opts=search_opts)
        self.assertEqual(self.columns, cmd_columns)
        data = self._get_data()
        self.assertEqual(data, tuple(cmd_data))

    def test_share_list_status(self):
        arglist = ['--status', self.new_share.status]
        verifylist = [('long', False), ('all_projects', False), ('name', None), ('status', self.new_share.status)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cmd_columns, cmd_data = self.cmd.take_action(parsed_args)
        search_opts = self._get_search_opts()
        search_opts['status'] = self.new_share.status
        self.shares_mock.list.assert_called_once_with(search_opts=search_opts)
        self.assertEqual(self.columns, cmd_columns)
        data = self._get_data()
        self.assertEqual(data, tuple(cmd_data))

    def test_share_list_all_tenants(self):
        arglist = ['--all-projects']
        verifylist = [('long', False), ('all_projects', True), ('name', None), ('status', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cmd_columns, cmd_data = self.cmd.take_action(parsed_args)
        search_opts = self._get_search_opts()
        search_opts['all_tenants'] = True
        self.shares_mock.list.assert_called_once_with(search_opts=search_opts)
        self.assertEqual(self.columns, cmd_columns)
        data = self._get_data()
        self.assertEqual(data, tuple(cmd_data))

    def test_share_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True), ('all_projects', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cmd_columns, cmd_data = self.cmd.take_action(parsed_args)
        search_opts = self._get_search_opts()
        self.shares_mock.list.assert_called_once_with(search_opts=search_opts)
        collist = ['ID', 'Name', 'Size', 'Share Protocol', 'Status', 'Is Public', 'Share Type Name', 'Availability Zone', 'Description', 'Share Network ID', 'Share Server ID', 'Share Type', 'Share Group ID', 'Host', 'User ID', 'Project ID', 'Access Rules Status', 'Source Snapshot ID', 'Supports Creating Snapshots', 'Supports Cloning Snapshots', 'Supports Mounting snapshots', 'Supports Reverting to Snapshot', 'Migration Task Status', 'Source Share Group Snapshot Member ID', 'Replication Type', 'Has Replicas', 'Created At', 'Properties']
        self.assertEqual(collist, cmd_columns)
        data = ((self.new_share.id, self.new_share.name, self.new_share.size, self.new_share.share_proto, self.new_share.status, self.new_share.is_public, self.new_share.share_type_name, self.new_share.availability_zone, self.new_share.description, self.new_share.share_network_id, self.new_share.share_server_id, self.new_share.share_type, self.new_share.share_group_id, self.new_share.host, self.new_share.user_id, self.new_share.project_id, self.new_share.access_rules_status, self.new_share.snapshot_id, self.new_share.snapshot_support, self.new_share.create_share_from_snapshot_support, self.new_share.mount_snapshot_support, self.new_share.revert_to_snapshot_support, self.new_share.task_state, self.new_share.source_share_group_snapshot_member_id, self.new_share.replication_type, self.new_share.has_replicas, self.new_share.created_at, self.new_share.metadata),)
        self.assertEqual(data, tuple(cmd_data))

    def test_share_list_with_marker_and_limit(self):
        arglist = ['--marker', self.new_share.id, '--limit', '2']
        verifylist = [('long', False), ('all_projects', False), ('name', None), ('status', None), ('marker', self.new_share.id), ('limit', 2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cmd_columns, cmd_data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, cmd_columns)
        search_opts = self._get_search_opts()
        search_opts['limit'] = 2
        search_opts['offset'] = self.new_share.id
        data = self._get_data()
        self.shares_mock.list.assert_called_once_with(search_opts=search_opts)
        self.assertEqual(data, tuple(cmd_data))

    def test_share_list_negative_limit(self):
        arglist = ['--limit', '-2']
        verifylist = [('limit', -2)]
        self.assertRaises(argparse.ArgumentTypeError, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_list_name_description_filter(self):
        arglist = ['--name~', self.new_share.name, '--description~', self.new_share.description]
        verifylist = [('name~', self.new_share.name), ('description~', self.new_share.description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cmd_columns, cmd_data = self.cmd.take_action(parsed_args)
        search_opts = self._get_search_opts()
        search_opts['name~'] = self.new_share.name
        search_opts['description~'] = self.new_share.description
        self.shares_mock.list.assert_called_once_with(search_opts=search_opts)
        self.assertEqual(self.columns, cmd_columns)
        data = self._get_data()
        self.assertEqual(data, tuple(cmd_data))

    def test_list_share_soft_deleted_api_version_exception(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.60')
        arglist = ['--soft-deleted']
        verifylist = [('soft_deleted', True)]
        search_opts = self._get_search_opts()
        search_opts['is_soft_deleted'] = True
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_list_share_api_version_exception(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.35')
        arglist = ['--name~', 'Name', '--description~', 'Description']
        verifylist = [('name~', 'Name'), ('description~', 'Description')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)