from openstack import exceptions
from openstack.tests.functional.baremetal import base
class TestBareMetalChassis(base.BaseBaremetalTest):

    def test_chassis_create_get_delete(self):
        chassis = self.create_chassis()
        loaded = self.conn.baremetal.get_chassis(chassis.id)
        self.assertEqual(loaded.id, chassis.id)
        self.conn.baremetal.delete_chassis(chassis, ignore_missing=False)
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_chassis, chassis.id)

    def test_chassis_update(self):
        chassis = self.create_chassis()
        chassis.extra = {'answer': 42}
        chassis = self.conn.baremetal.update_chassis(chassis)
        self.assertEqual({'answer': 42}, chassis.extra)
        chassis = self.conn.baremetal.get_chassis(chassis.id)
        self.assertEqual({'answer': 42}, chassis.extra)

    def test_chassis_patch(self):
        chassis = self.create_chassis()
        chassis = self.conn.baremetal.patch_chassis(chassis, dict(path='/extra/answer', op='add', value=42))
        self.assertEqual({'answer': 42}, chassis.extra)
        chassis = self.conn.baremetal.get_chassis(chassis.id)
        self.assertEqual({'answer': 42}, chassis.extra)

    def test_chassis_negative_non_existing(self):
        uuid = '5c9dcd04-2073-49bc-9618-99ae634d8971'
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_chassis, uuid)
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.find_chassis, uuid, ignore_missing=False)
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.delete_chassis, uuid, ignore_missing=False)
        self.assertIsNone(self.conn.baremetal.find_chassis(uuid))
        self.assertIsNone(self.conn.baremetal.delete_chassis(uuid))