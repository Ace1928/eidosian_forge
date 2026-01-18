import argparse
from unittest import mock
import uuid
from osc_lib import exceptions
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient.osc import utils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_groups as osc_share_groups
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareGroupList(TestShareGroup):
    columns = ['id', 'name', 'status', 'description']
    column_headers = utils.format_column_headers(columns)

    def setUp(self):
        super(TestShareGroupList, self).setUp()
        self.new_share_group = manila_fakes.FakeShareGroup.create_one_share_group()
        self.groups_mock.list.return_value = [self.new_share_group]
        self.share_group = manila_fakes.FakeShareGroup.create_one_share_group()
        self.groups_mock.get.return_value = self.share_group
        self.share_groups_list = manila_fakes.FakeShareGroup.create_share_groups(count=2)
        self.groups_mock.list.return_value = self.share_groups_list
        self.values = (oscutils.get_dict_properties(s._info, self.columns) for s in self.share_groups_list)
        self.cmd = osc_share_groups.ListShareGroup(self.app, None)

    def test_share_group_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.groups_mock.list.assert_called_with(search_opts={'all_tenants': False, 'name': None, 'status': None, 'share_server_id': None, 'share_group_type': None, 'snapshot': None, 'host': None, 'share_network': None, 'project_id': None, 'limit': None, 'offset': None, 'name~': None, 'description~': None, 'description': None})
        self.assertEqual(self.column_headers, columns)
        self.assertEqual(list(self.values), list(data))

    def test_list_share_group_api_version_exception(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.35')
        arglist = ['--description', 'Description']
        verifylist = [('description', 'Description')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_list_share_groups_all_projects(self):
        all_tenants_list = self.column_headers.copy()
        all_tenants_list.append('Project ID')
        list_values = (oscutils.get_dict_properties(s._info, all_tenants_list) for s in self.share_groups_list)
        arglist = ['--all-projects']
        verifylist = [('all_projects', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.groups_mock.list.assert_called_with(search_opts={'all_tenants': True, 'name': None, 'status': None, 'share_server_id': None, 'share_group_type': None, 'snapshot': None, 'host': None, 'share_network': None, 'project_id': None, 'limit': None, 'offset': None, 'name~': None, 'description~': None, 'description': None})
        self.assertEqual(all_tenants_list, columns)
        self.assertEqual(list(list_values), list(data))

    def test_share_group_list_name(self):
        arglist = ['--name', self.new_share_group.name]
        verifylist = [('name', self.new_share_group.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': False, 'name': None, 'status': None, 'share_server_id': None, 'share_group_type': None, 'snapshot': None, 'host': None, 'share_network': None, 'project_id': None, 'limit': None, 'offset': None, 'name~': None, 'description~': None, 'description': None}
        search_opts['name'] = self.new_share_group.name
        self.groups_mock.list.assert_called_once_with(search_opts=search_opts)
        self.assertEqual(self.column_headers, columns)
        self.assertEqual(list(self.values), list(data))

    def test_share_group_list_description(self):
        arglist = ['--description', self.new_share_group.description]
        verifylist = [('description', self.new_share_group.description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': False, 'name': None, 'status': None, 'share_server_id': None, 'share_group_type': None, 'snapshot': None, 'host': None, 'share_network': None, 'project_id': None, 'limit': None, 'offset': None, 'name~': None, 'description~': None, 'description': None}
        search_opts['description'] = self.new_share_group.description
        self.groups_mock.list.assert_called_once_with(search_opts=search_opts)
        self.assertEqual(self.column_headers, columns)
        self.assertEqual(list(self.values), list(data))

    def test_share_group_list_status(self):
        arglist = ['--status', self.new_share_group.status]
        verifylist = [('status', self.new_share_group.status)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': False, 'name': None, 'status': None, 'share_server_id': None, 'share_group_type': None, 'snapshot': None, 'host': None, 'share_network': None, 'project_id': None, 'limit': None, 'offset': None, 'name~': None, 'description~': None, 'description': None}
        search_opts['status'] = self.new_share_group.status
        self.groups_mock.list.assert_called_once_with(search_opts=search_opts)
        self.assertEqual(self.column_headers, columns)
        self.assertEqual(list(self.values), list(data))

    def test_share_group_list_marker_and_limit(self):
        arglist = ['--marker', self.new_share_group.id, '--limit', '2']
        verifylist = [('marker', self.new_share_group.id), ('limit', 2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': False, 'name': None, 'status': None, 'share_server_id': None, 'share_group_type': None, 'snapshot': None, 'host': None, 'share_network': None, 'project_id': None, 'limit': 2, 'offset': self.new_share_group.id, 'name~': None, 'description~': None, 'description': None}
        self.groups_mock.list.assert_called_once_with(search_opts=search_opts)
        self.assertEqual(self.column_headers, columns)
        self.assertEqual(list(self.values), list(data))

    def test_share_group_list_negative_limit(self):
        arglist = ['--limit', '-2']
        verifylist = [('limit', -2)]
        self.assertRaises(argparse.ArgumentTypeError, self.check_parser, self.cmd, arglist, verifylist)