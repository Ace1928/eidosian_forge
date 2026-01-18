from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
class TestBackupRestore(TestBackup):
    volume = volume_fakes.create_one_volume()
    backup = volume_fakes.create_one_backup(attrs={'volume_id': volume.id})

    def setUp(self):
        super().setUp()
        self.volume_sdk_client.find_backup.return_value = self.backup
        self.volume_sdk_client.find_volume.return_value = self.volume
        self.volume_sdk_client.restore_backup.return_value = volume_fakes.create_one_volume({'id': self.volume['id']})
        self.cmd = volume_backup.RestoreVolumeBackup(self.app, None)

    def test_backup_restore(self):
        self.volume_sdk_client.find_volume.side_effect = exceptions.CommandError()
        arglist = [self.backup.id]
        verifylist = [('backup', self.backup.id), ('volume', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volume_sdk_client.restore_backup.assert_called_with(self.backup.id, volume_id=None, name=None)
        self.assertIsNotNone(result)

    def test_backup_restore_with_volume(self):
        self.volume_sdk_client.find_volume.side_effect = exceptions.CommandError()
        arglist = [self.backup.id, self.backup.volume_id]
        verifylist = [('backup', self.backup.id), ('volume', self.backup.volume_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volume_sdk_client.restore_backup.assert_called_with(self.backup.id, volume_id=None, name=self.backup.volume_id)
        self.assertIsNotNone(result)

    def test_backup_restore_with_volume_force(self):
        arglist = ['--force', self.backup.id, self.volume.name]
        verifylist = [('force', True), ('backup', self.backup.id), ('volume', self.volume.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volume_sdk_client.restore_backup.assert_called_with(self.backup.id, volume_id=self.volume.id, name=None)
        self.assertIsNotNone(result)

    def test_backup_restore_with_volume_existing(self):
        arglist = [self.backup.id, self.volume.name]
        verifylist = [('backup', self.backup.id), ('volume', self.volume.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)