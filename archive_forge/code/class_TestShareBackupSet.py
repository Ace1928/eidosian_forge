from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.osc.v2 import share_backups as osc_share_backups
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareBackupSet(TestShareBackup):

    def setUp(self):
        super(TestShareBackupSet, self).setUp()
        self.share_backup = manila_fakes.FakeShareBackup.create_one_backup()
        self.backups_mock.get.return_value = self.share_backup
        self.cmd = osc_share_backups.SetShareBackup(self.app, None)

    def test_set_share_backup_name(self):
        arglist = [self.share_backup.id, '--name', 'FAKE_SHARE_BACKUP_NAME']
        verifylist = [('backup', self.share_backup.id), ('name', 'FAKE_SHARE_BACKUP_NAME')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.backups_mock.update.assert_called_with(self.share_backup, name=parsed_args.name)
        self.assertIsNone(result)

    def test_set_backup_status(self):
        arglist = [self.share_backup.id, '--status', 'available']
        verifylist = [('backup', self.share_backup.id), ('status', 'available')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.backups_mock.reset_status.assert_called_with(self.share_backup, parsed_args.status)
        self.assertIsNone(result)