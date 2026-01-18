from unittest import mock
from keystoneauth1 import adapter
from openstack.shared_file_system.v2 import share
from openstack.tests.unit import base
class TestShares(base.TestCase):

    def test_basic(self):
        shares_resource = share.Share()
        self.assertEqual('shares', shares_resource.resources_key)
        self.assertEqual('/shares', shares_resource.base_path)
        self.assertTrue(shares_resource.allow_list)
        self.assertTrue(shares_resource.allow_create)
        self.assertTrue(shares_resource.allow_fetch)
        self.assertTrue(shares_resource.allow_commit)
        self.assertTrue(shares_resource.allow_delete)

    def test_make_shares(self):
        shares_resource = share.Share(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], shares_resource.id)
        self.assertEqual(EXAMPLE['size'], shares_resource.size)
        self.assertEqual(EXAMPLE['availability_zone'], shares_resource.availability_zone)
        self.assertEqual(EXAMPLE['created_at'], shares_resource.created_at)
        self.assertEqual(EXAMPLE['status'], shares_resource.status)
        self.assertEqual(EXAMPLE['name'], shares_resource.name)
        self.assertEqual(EXAMPLE['description'], shares_resource.description)
        self.assertEqual(EXAMPLE['project_id'], shares_resource.project_id)
        self.assertEqual(EXAMPLE['snapshot_id'], shares_resource.snapshot_id)
        self.assertEqual(EXAMPLE['share_network_id'], shares_resource.share_network_id)
        self.assertEqual(EXAMPLE['share_protocol'], shares_resource.share_protocol)
        self.assertEqual(EXAMPLE['metadata'], shares_resource.metadata)
        self.assertEqual(EXAMPLE['share_type'], shares_resource.share_type)
        self.assertEqual(EXAMPLE['is_public'], shares_resource.is_public)
        self.assertEqual(EXAMPLE['is_snapshot_supported'], shares_resource.is_snapshot_supported)
        self.assertEqual(EXAMPLE['task_state'], shares_resource.task_state)
        self.assertEqual(EXAMPLE['share_type_name'], shares_resource.share_type_name)
        self.assertEqual(EXAMPLE['access_rules_status'], shares_resource.access_rules_status)
        self.assertEqual(EXAMPLE['replication_type'], shares_resource.replication_type)
        self.assertEqual(EXAMPLE['is_replicated'], shares_resource.is_replicated)
        self.assertEqual(EXAMPLE['user_id'], shares_resource.user_id)
        self.assertEqual(EXAMPLE['is_creating_new_share_from_snapshot_supported'], shares_resource.is_creating_new_share_from_snapshot_supported)
        self.assertEqual(EXAMPLE['is_reverting_to_snapshot_supported'], shares_resource.is_reverting_to_snapshot_supported)
        self.assertEqual(EXAMPLE['share_group_id'], shares_resource.share_group_id)
        self.assertEqual(EXAMPLE['source_share_group_snapshot_member_id'], shares_resource.source_share_group_snapshot_member_id)
        self.assertEqual(EXAMPLE['is_mounting_snapshot_supported'], shares_resource.is_mounting_snapshot_supported)
        self.assertEqual(EXAMPLE['progress'], shares_resource.progress)
        self.assertEqual(EXAMPLE['share_server_id'], shares_resource.share_server_id)
        self.assertEqual(EXAMPLE['host'], shares_resource.host)