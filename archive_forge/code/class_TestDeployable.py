import uuid
from openstack.accelerator.v2 import deployable
from openstack.tests.unit import base
class TestDeployable(base.TestCase):

    def test_basic(self):
        sot = deployable.Deployable()
        self.assertEqual('deployable', sot.resource_key)
        self.assertEqual('deployables', sot.resources_key)
        self.assertEqual('/deployables', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = deployable.Deployable(**EXAMPLE)
        self.assertEqual(EXAMPLE['uuid'], sot.id)
        self.assertEqual(EXAMPLE['parent_id'], sot.parent_id)
        self.assertEqual(EXAMPLE['root_id'], sot.root_id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['num_accelerators'], sot.num_accelerators)
        self.assertEqual(EXAMPLE['device_id'], sot.device_id)
        self.assertEqual(EXAMPLE['created_at'], sot.created_at)
        self.assertEqual(EXAMPLE['updated_at'], sot.updated_at)