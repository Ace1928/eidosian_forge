from cinderclient.tests.functional import base
class CinderBackupTests(base.ClientTestBase):
    """Check of base cinder backup commands."""
    BACKUP_PROPERTY = ('id', 'name', 'volume_id')

    def test_backup_create_and_delete(self):
        """Create a volume backup and then delete."""
        volume = self.object_create('volume', params='1')
        backup = self.object_create('backup', params=volume['id'])
        self.assert_object_details(self.BACKUP_PROPERTY, backup.keys())
        self.object_delete('volume', volume['id'])
        self.check_object_deleted('volume', volume['id'])
        self.object_delete('backup', backup['id'])
        self.check_object_deleted('backup', backup['id'])