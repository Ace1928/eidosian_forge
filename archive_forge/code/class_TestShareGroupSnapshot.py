from unittest import mock
from keystoneauth1 import adapter
from openstack.shared_file_system.v2 import share_group_snapshot
from openstack.tests.unit import base
class TestShareGroupSnapshot(base.TestCase):

    def test_basic(self):
        share_group_snapshots = share_group_snapshot.ShareGroupSnapshot()
        self.assertEqual('share_group_snapshot', share_group_snapshots.resource_key)
        self.assertEqual('share_group_snapshots', share_group_snapshots.resources_key)
        self.assertEqual('/share-group-snapshots', share_group_snapshots.base_path)
        self.assertTrue(share_group_snapshots.allow_create)
        self.assertTrue(share_group_snapshots.allow_fetch)
        self.assertTrue(share_group_snapshots.allow_commit)
        self.assertTrue(share_group_snapshots.allow_delete)
        self.assertTrue(share_group_snapshots.allow_list)
        self.assertFalse(share_group_snapshots.allow_head)

    def test_make_share_groups(self):
        share_group_snapshots = share_group_snapshot.ShareGroupSnapshot(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], share_group_snapshots.id)
        self.assertEqual(EXAMPLE['name'], share_group_snapshots.name)
        self.assertEqual(EXAMPLE['created_at'], share_group_snapshots.created_at)
        self.assertEqual(EXAMPLE['status'], share_group_snapshots.status)
        self.assertEqual(EXAMPLE['description'], share_group_snapshots.description)
        self.assertEqual(EXAMPLE['project_id'], share_group_snapshots.project_id)
        self.assertEqual(EXAMPLE['share_group_id'], share_group_snapshots.share_group_id)