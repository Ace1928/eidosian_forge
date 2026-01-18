from openstack import exceptions
from openstack.tests.functional.baremetal import base
class TestBareMetalDriver(base.BaseBaremetalTest):

    def test_fake_hardware_get(self):
        driver = self.conn.baremetal.get_driver('fake-hardware')
        self.assertEqual('fake-hardware', driver.name)
        self.assertNotEqual([], driver.hosts)

    def test_fake_hardware_list(self):
        drivers = self.conn.baremetal.drivers()
        self.assertIn('fake-hardware', [d.name for d in drivers])

    def test_driver_negative_non_existing(self):
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_driver, 'not-a-driver')