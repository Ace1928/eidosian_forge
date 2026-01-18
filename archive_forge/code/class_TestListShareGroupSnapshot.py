import logging
import uuid
from osc_lib import exceptions
from osc_lib import utils as oscutils
from unittest import mock
from manilaclient import api_versions
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestListShareGroupSnapshot(TestShareGroupSnapshot):
    columns = ['ID', 'Name', 'Status', 'Description']

    def setUp(self):
        super(TestListShareGroupSnapshot, self).setUp()
        self.share_group = manila_fakes.FakeShareGroup.create_one_share_group()
        self.groups_mock.get.return_value = self.share_group
        self.share_group_snapshot = manila_fakes.FakeShareGroupSnapshot.create_one_share_group_snapshot({'share_group_id': self.share_group.id})
        self.share_group_snapshots_list = [self.share_group_snapshot]
        self.group_snapshot_mocks.list.return_value = self.share_group_snapshots_list
        self.values = (oscutils.get_dict_properties(s._info, self.columns) for s in self.share_group_snapshots_list)
        self.cmd = osc_share_group_snapshots.ListShareGroupSnapshot(self.app, None)

    def test_share_group_snapshot_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.group_snapshot_mocks.list.assert_called_once_with(search_opts={'all_tenants': False, 'name': None, 'status': None, 'share_group_id': None, 'limit': None, 'offset': None})
        self.assertEqual(self.columns, columns)
        self.assertEqual(list(self.values), list(data))

    def test_share_group_snapshot_list_detail_all_projects(self):
        columns_detail = ['ID', 'Name', 'Status', 'Description', 'Created At', 'Share Group ID', 'Project ID']
        values = (oscutils.get_dict_properties(s._info, columns_detail) for s in self.share_group_snapshots_list)
        arglist = ['--detailed', '--all-projects']
        verifylist = [('detailed', True), ('all_projects', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.group_snapshot_mocks.list.assert_called_once_with(search_opts={'all_tenants': True, 'name': None, 'status': None, 'share_group_id': None, 'limit': None, 'offset': None})
        self.assertEqual(columns_detail, columns)
        self.assertEqual(list(values), list(data))

    def test_share_group_snapshot_list_search_options(self):
        arglist = ['--name', self.share_group_snapshot.name, '--status', self.share_group_snapshot.status, '--share-group', self.share_group.id]
        verifylist = [('name', self.share_group_snapshot.name), ('status', self.share_group_snapshot.status), ('share_group', self.share_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.groups_mock.get.assert_called_with(self.share_group.id)
        self.group_snapshot_mocks.list.assert_called_once_with(search_opts={'all_tenants': False, 'name': self.share_group_snapshot.name, 'status': self.share_group_snapshot.status, 'share_group_id': self.share_group.id, 'limit': None, 'offset': None})
        self.assertEqual(self.columns, columns)
        self.assertEqual(list(self.values), list(data))