from openstack.baremetal.v1 import volume_target
from openstack.tests.unit import base
class TestVolumeTarget(base.TestCase):

    def test_basic(self):
        sot = volume_target.VolumeTarget()
        self.assertIsNone(sot.resource_key)
        self.assertEqual('targets', sot.resources_key)
        self.assertEqual('/volume/targets', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertEqual('PATCH', sot.commit_method)

    def test_instantiate(self):
        sot = volume_target.VolumeTarget(**FAKE)
        self.assertEqual(FAKE['boot_index'], sot.boot_index)
        self.assertEqual(FAKE['created_at'], sot.created_at)
        self.assertEqual(FAKE['extra'], sot.extra)
        self.assertEqual(FAKE['links'], sot.links)
        self.assertEqual(FAKE['node_uuid'], sot.node_id)
        self.assertEqual(FAKE['properties'], sot.properties)
        self.assertEqual(FAKE['updated_at'], sot.updated_at)
        self.assertEqual(FAKE['uuid'], sot.id)
        self.assertEqual(FAKE['volume_id'], sot.volume_id)
        self.assertEqual(FAKE['volume_type'], sot.volume_type)