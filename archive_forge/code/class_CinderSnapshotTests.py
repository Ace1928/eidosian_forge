from cinderclient.tests.functional import base
class CinderSnapshotTests(base.ClientTestBase):
    """Check of base cinder snapshot commands."""
    SNAPSHOT_PROPERTY = ('created_at', 'description', 'metadata', 'id', 'name', 'size', 'status', 'volume_id')

    def test_snapshot_create_and_delete(self):
        """Create a volume snapshot and then delete."""
        volume = self.object_create('volume', params='1')
        snapshot = self.object_create('snapshot', params=volume['id'])
        self.assert_object_details(self.SNAPSHOT_PROPERTY, snapshot.keys())
        self.object_delete('snapshot', snapshot['id'])
        self.check_object_deleted('snapshot', snapshot['id'])
        self.object_delete('volume', volume['id'])
        self.check_object_deleted('volume', volume['id'])