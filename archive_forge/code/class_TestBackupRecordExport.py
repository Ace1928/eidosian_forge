from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import backup_record
class TestBackupRecordExport(TestBackupRecord):
    new_backup = volume_fakes.create_one_backup(attrs={'volume_id': 'a54708a2-0388-4476-a909-09579f885c25'})
    new_record = volume_fakes.create_backup_record()

    def setUp(self):
        super().setUp()
        self.backups_mock.export_record.return_value = self.new_record
        self.backups_mock.get.return_value = self.new_backup
        self.cmd = backup_record.ExportBackupRecord(self.app, None)

    def test_backup_export_table(self):
        arglist = [self.new_backup.name]
        verifylist = [('backup', self.new_backup.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        parsed_args.formatter = 'table'
        columns, __ = self.cmd.take_action(parsed_args)
        self.backups_mock.export_record.assert_called_with(self.new_backup.id)
        expected_columns = ('Backup Service', 'Metadata')
        self.assertEqual(columns, expected_columns)

    def test_backup_export_json(self):
        arglist = [self.new_backup.name]
        verifylist = [('backup', self.new_backup.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        parsed_args.formatter = 'json'
        columns, __ = self.cmd.take_action(parsed_args)
        self.backups_mock.export_record.assert_called_with(self.new_backup.id)
        expected_columns = ('backup_service', 'backup_url')
        self.assertEqual(columns, expected_columns)