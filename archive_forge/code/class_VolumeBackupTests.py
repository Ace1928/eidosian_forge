import uuid
from openstackclient.tests.functional.volume.v2 import common
class VolumeBackupTests(common.BaseVolumeTests):
    """Functional tests for volume backups."""

    def setUp(self):
        super(VolumeBackupTests, self).setUp()
        self.backup_enabled = False
        serv_list = self.openstack('volume service list', parse_output=True)
        for service in serv_list:
            if service['Binary'] == 'cinder-backup':
                if service['Status'] == 'enabled':
                    self.backup_enabled = True

    def test_volume_backup_restore(self):
        """Test restore backup"""
        if not self.backup_enabled:
            self.skipTest('Backup service is not enabled')
        vol_id = uuid.uuid4().hex
        self.openstack('volume create ' + '--size 1 ' + vol_id, parse_output=True)
        self.wait_for_status('volume', vol_id, 'available')
        backup = self.openstack('volume backup create ' + vol_id, parse_output=True)
        self.wait_for_status('volume backup', backup['id'], 'available')
        backup_restored = self.openstack('volume backup restore %s %s' % (backup['id'], vol_id), parse_output=True)
        self.assertEqual(backup_restored['backup_id'], backup['id'])
        self.wait_for_status('volume backup', backup['id'], 'available')
        self.wait_for_status('volume', backup_restored['volume_id'], 'available')
        self.addCleanup(self.openstack, 'volume delete %s' % vol_id)