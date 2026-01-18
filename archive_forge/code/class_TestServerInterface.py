from openstack.compute.v2 import server_diagnostics
from openstack.tests.unit import base
class TestServerInterface(base.TestCase):

    def test_basic(self):
        sot = server_diagnostics.ServerDiagnostics()
        self.assertEqual('diagnostics', sot.resource_key)
        self.assertEqual('/servers/%(server_id)s/diagnostics', sot.base_path)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.requires_id)

    def test_make_it(self):
        sot = server_diagnostics.ServerDiagnostics(**EXAMPLE)
        self.assertEqual(EXAMPLE['config_drive'], sot.has_config_drive)
        self.assertEqual(EXAMPLE['cpu_details'], sot.cpu_details)
        self.assertEqual(EXAMPLE['disk_details'], sot.disk_details)
        self.assertEqual(EXAMPLE['driver'], sot.driver)
        self.assertEqual(EXAMPLE['hypervisor'], sot.hypervisor)
        self.assertEqual(EXAMPLE['hypervisor_os'], sot.hypervisor_os)
        self.assertEqual(EXAMPLE['memory_details'], sot.memory_details)
        self.assertEqual(EXAMPLE['nic_details'], sot.nic_details)
        self.assertEqual(EXAMPLE['num_cpus'], sot.num_cpus)
        self.assertEqual(EXAMPLE['num_disks'], sot.num_disks)
        self.assertEqual(EXAMPLE['num_nics'], sot.num_nics)
        self.assertEqual(EXAMPLE['state'], sot.state)
        self.assertEqual(EXAMPLE['uptime'], sot.uptime)