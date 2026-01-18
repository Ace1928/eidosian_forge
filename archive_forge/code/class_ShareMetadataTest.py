from openstack.shared_file_system.v2 import share as _share
from openstack.tests.functional.shared_file_system import base
class ShareMetadataTest(base.BaseSharedFileSystemTest):

    def setUp(self):
        super().setUp()
        self.SHARE_NAME = self.getUniqueString()
        my_share = self.create_share(name=self.SHARE_NAME, size=2, share_type='dhss_false', share_protocol='NFS', description=None)
        self.SHARE_ID = my_share.id
        self.assertIsNotNone(my_share)
        self.assertIsNotNone(my_share.id)

    def test_create(self):
        meta = {'foo': 'bar'}
        created_share = self.user_cloud.shared_file_system.create_share_metadata(self.SHARE_ID, **meta)
        assert isinstance(created_share, _share.Share)
        self.assertEqual(created_share['metadata'], meta)

    def test_get_item(self):
        meta = {'foo': 'bar'}
        created_share = self.user_cloud.shared_file_system.create_share_metadata(self.SHARE_ID, **meta)
        returned_share = self.user_cloud.shared_file_system.get_share_metadata_item(self.SHARE_ID, 'foo')
        self.assertEqual(created_share['metadata']['foo'], returned_share['metadata']['foo'])

    def test_get(self):
        meta = {'foo': 'bar'}
        created_share = self.user_cloud.shared_file_system.create_share_metadata(self.SHARE_ID, **meta)
        returned_share = self.user_cloud.shared_file_system.get_share_metadata(self.SHARE_ID)
        self.assertEqual(created_share['metadata']['foo'], returned_share['metadata']['foo'])

    def test_update(self):
        meta = {'foo': 'bar'}
        created_share = self.user_cloud.shared_file_system.create_share_metadata(self.SHARE_ID, **meta)
        new_meta = {'newFoo': 'newBar'}
        full_meta = {'foo': 'bar', 'newFoo': 'newBar'}
        empty_meta = {}
        updated_share = self.user_cloud.shared_file_system.update_share_metadata(created_share, new_meta)
        self.assertEqual(updated_share['metadata'], new_meta)
        full_metadata = self.user_cloud.shared_file_system.get_share_metadata(created_share)['metadata']
        self.assertEqual(full_metadata, full_meta)
        share_with_deleted_metadata = self.user_cloud.shared_file_system.update_share_metadata(updated_share, empty_meta)
        self.assertEqual(share_with_deleted_metadata['metadata'], empty_meta)

    def test_delete(self):
        meta = {'foo': 'bar', 'newFoo': 'newBar'}
        created_share = self.user_cloud.shared_file_system.create_share_metadata(self.SHARE_ID, **meta)
        self.user_cloud.shared_file_system.delete_share_metadata(created_share, ['foo', 'invalidKey'])
        deleted_share = self.user_cloud.shared_file_system.get_share_metadata(self.SHARE_ID)
        self.assertEqual(deleted_share['metadata'], {'newFoo': 'newBar'})