from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.osc.v2 import share_backups as osc_share_backups
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareBackupList(TestShareBackup):
    columns = ['ID', 'Name', 'Share ID', 'Status']
    detailed_columns = ['ID', 'Name', 'Share ID', 'Status', 'Description', 'Size', 'Created At', 'Updated At', 'Availability Zone', 'Progress', 'Restore Progress', 'Host', 'Topic']

    def setUp(self):
        super(TestShareBackupList, self).setUp()
        self.share = manila_fakes.FakeShare.create_one_share()
        self.shares_mock.get.return_value = self.share
        self.backups_list = manila_fakes.FakeShareBackup.create_share_backups(count=2)
        self.backups_mock.list.return_value = self.backups_list
        self.values = (oscutils.get_dict_properties(i._info, self.columns) for i in self.backups_list)
        self.detailed_values = (oscutils.get_dict_properties(i._info, self.detailed_columns) for i in self.backups_list)
        self.cmd = osc_share_backups.ListShareBackup(self.app, None)

    def test_share_backup_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.backups_mock.list.assert_called_with(detailed=0, search_opts={'offset': None, 'limit': None, 'name': None, 'description': None, 'name~': None, 'description~': None, 'status': None, 'share_id': None}, sort_key=None, sort_dir=None)
        self.assertEqual(self.columns, columns)
        self.assertEqual(list(self.values), list(data))

    def test_share_backup_list_detail(self):
        arglist = ['--detail']
        verifylist = [('detail', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.backups_mock.list.assert_called_with(detailed=1, search_opts={'offset': None, 'limit': None, 'name': None, 'description': None, 'name~': None, 'description~': None, 'status': None, 'share_id': None}, sort_key=None, sort_dir=None)
        self.assertEqual(self.detailed_columns, columns)
        self.assertEqual(list(self.detailed_values), list(data))

    def test_share_backup_list_for_share(self):
        arglist = ['--share', self.share.id]
        verifylist = [('share', self.share.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.backups_mock.list.assert_called_with(detailed=0, search_opts={'offset': None, 'limit': None, 'name': None, 'description': None, 'name~': None, 'description~': None, 'status': None, 'share_id': self.share.id}, sort_key=None, sort_dir=None)
        self.assertEqual(self.columns, columns)
        self.assertEqual(list(self.values), list(data))