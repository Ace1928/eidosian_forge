from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.osc.v2 import share_backups as osc_share_backups
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareBackupShow(TestShareBackup):

    def setUp(self):
        super(TestShareBackupShow, self).setUp()
        self.share_backup = manila_fakes.FakeShareBackup.create_one_backup()
        self.backups_mock.get.return_value = self.share_backup
        self.cmd = osc_share_backups.ShowShareBackup(self.app, None)
        self.data = tuple(self.share_backup._info.values())
        self.columns = tuple(self.share_backup._info.keys())

    def test_share_backup_show_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_backup_show(self):
        arglist = [self.share_backup.id]
        verifylist = [('backup', self.share_backup.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.backups_mock.get.assert_called_with(self.share_backup.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)