from openstack.cloud import meta
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
class TestDeleteVolumeSnapshot(base.TestCase):

    def setUp(self):
        super(TestDeleteVolumeSnapshot, self).setUp()
        self.use_cinder()

    def test_delete_volume_snapshot(self):
        """
        Test that delete_volume_snapshot without a wait returns True instance
        when the volume snapshot deletes.
        """
        fake_snapshot = fakes.FakeVolumeSnapshot('1234', 'available', 'foo', 'derpysnapshot')
        fake_snapshot_dict = meta.obj_to_munch(fake_snapshot)
        self.register_uris([dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['snapshots', 'detail']), json={'snapshots': [fake_snapshot_dict]}), dict(method='DELETE', uri=self.get_mock_url('volumev3', 'public', append=['snapshots', fake_snapshot_dict['id']]))])
        self.assertTrue(self.cloud.delete_volume_snapshot(name_or_id='1234', wait=False))
        self.assert_calls()

    def test_delete_volume_snapshot_with_error(self):
        """
        Test that a exception while deleting a volume snapshot will cause an
        SDKException.
        """
        fake_snapshot = fakes.FakeVolumeSnapshot('1234', 'available', 'foo', 'derpysnapshot')
        fake_snapshot_dict = meta.obj_to_munch(fake_snapshot)
        self.register_uris([dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['snapshots', 'detail']), json={'snapshots': [fake_snapshot_dict]}), dict(method='DELETE', uri=self.get_mock_url('volumev3', 'public', append=['snapshots', fake_snapshot_dict['id']]), status_code=404)])
        self.assertRaises(exceptions.SDKException, self.cloud.delete_volume_snapshot, name_or_id='1234')
        self.assert_calls()

    def test_delete_volume_snapshot_with_timeout(self):
        """
        Test that a timeout while waiting for the volume snapshot to delete
        raises an exception in delete_volume_snapshot.
        """
        fake_snapshot = fakes.FakeVolumeSnapshot('1234', 'available', 'foo', 'derpysnapshot')
        fake_snapshot_dict = meta.obj_to_munch(fake_snapshot)
        self.register_uris([dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['snapshots', 'detail']), json={'snapshots': [fake_snapshot_dict]}), dict(method='DELETE', uri=self.get_mock_url('volumev3', 'public', append=['snapshots', fake_snapshot_dict['id']])), dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['snapshots', '1234']), json={'snapshot': fake_snapshot_dict})])
        self.assertRaises(exceptions.ResourceTimeout, self.cloud.delete_volume_snapshot, name_or_id='1234', wait=True, timeout=0.01)
        self.assert_calls(do_count=False)