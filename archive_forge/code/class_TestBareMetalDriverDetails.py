from openstack import exceptions
from openstack.tests.functional.baremetal import base
class TestBareMetalDriverDetails(base.BaseBaremetalTest):
    min_microversion = '1.30'

    def test_fake_hardware_get(self):
        driver = self.conn.baremetal.get_driver('fake-hardware')
        self.assertEqual('fake-hardware', driver.name)
        for iface in ('boot', 'deploy', 'management', 'power'):
            self.assertIn('fake', getattr(driver, 'enabled_%s_interfaces' % iface))
            self.assertEqual('fake', getattr(driver, 'default_%s_interface' % iface))
        self.assertNotEqual([], driver.hosts)

    def test_fake_hardware_list_details(self):
        drivers = self.conn.baremetal.drivers(details=True)
        driver = [d for d in drivers if d.name == 'fake-hardware'][0]
        for iface in ('boot', 'deploy', 'management', 'power'):
            self.assertIn('fake', getattr(driver, 'enabled_%s_interfaces' % iface))
            self.assertEqual('fake', getattr(driver, 'default_%s_interface' % iface))