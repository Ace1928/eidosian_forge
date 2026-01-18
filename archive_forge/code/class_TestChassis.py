from openstack.baremetal.v1 import chassis
from openstack.tests.unit import base
class TestChassis(base.TestCase):

    def test_basic(self):
        sot = chassis.Chassis()
        self.assertIsNone(sot.resource_key)
        self.assertEqual('chassis', sot.resources_key)
        self.assertEqual('/chassis', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertEqual('PATCH', sot.commit_method)

    def test_instantiate(self):
        sot = chassis.Chassis(**FAKE)
        self.assertEqual(FAKE['uuid'], sot.id)
        self.assertEqual(FAKE['created_at'], sot.created_at)
        self.assertEqual(FAKE['description'], sot.description)
        self.assertEqual(FAKE['extra'], sot.extra)
        self.assertEqual(FAKE['links'], sot.links)
        self.assertEqual(FAKE['nodes'], sot.nodes)
        self.assertEqual(FAKE['updated_at'], sot.updated_at)