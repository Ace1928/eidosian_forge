from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import backup_record
class TestBackupRecordImport(TestBackupRecord):
    new_backup = volume_fakes.create_one_backup(attrs={'volume_id': 'a54708a2-0388-4476-a909-09579f885c25'})
    new_import = volume_fakes.import_backup_record()

    def setUp(self):
        super().setUp()
        self.backups_mock.import_record.return_value = self.new_import
        self.cmd = backup_record.ImportBackupRecord(self.app, None)

    def test_backup_import(self):
        arglist = ['cinder.backup.drivers.swift.SwiftBackupDriver', 'fake_backup_record_data']
        verifylist = [('backup_service', 'cinder.backup.drivers.swift.SwiftBackupDriver'), ('backup_metadata', 'fake_backup_record_data')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, __ = self.cmd.take_action(parsed_args)
        self.backups_mock.import_record.assert_called_with('cinder.backup.drivers.swift.SwiftBackupDriver', 'fake_backup_record_data')
        self.assertEqual(columns, ('backup',))