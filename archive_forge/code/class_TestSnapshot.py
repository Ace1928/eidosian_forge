from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v2 import snapshot
from openstack.tests.unit import base
class TestSnapshot(base.TestCase):

    def test_basic(self):
        sot = snapshot.Snapshot(SNAPSHOT)
        self.assertEqual('snapshot', sot.resource_key)
        self.assertEqual('snapshots', sot.resources_key)
        self.assertEqual('/snapshots', sot.base_path)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertDictEqual({'name': 'name', 'status': 'status', 'all_projects': 'all_tenants', 'volume_id': 'volume_id', 'limit': 'limit', 'marker': 'marker'}, sot._query_mapping._mapping)

    def test_create_basic(self):
        sot = snapshot.Snapshot(**SNAPSHOT)
        self.assertEqual(SNAPSHOT['id'], sot.id)
        self.assertEqual(SNAPSHOT['status'], sot.status)
        self.assertEqual(SNAPSHOT['created_at'], sot.created_at)
        self.assertEqual(SNAPSHOT['updated_at'], sot.updated_at)
        self.assertEqual(SNAPSHOT['metadata'], sot.metadata)
        self.assertEqual(SNAPSHOT['volume_id'], sot.volume_id)
        self.assertEqual(SNAPSHOT['size'], sot.size)
        self.assertEqual(SNAPSHOT['name'], sot.name)
        self.assertTrue(sot.is_forced)