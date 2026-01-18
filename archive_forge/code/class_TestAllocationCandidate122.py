import uuid
from osc_placement.tests.functional import base
class TestAllocationCandidate122(base.BaseTestCase):
    VERSION = '1.22'

    def test_hide_forbidden_trait(self):
        rp1 = self.resource_provider_create()
        rp2 = self.resource_provider_create()
        rp3 = self.resource_provider_create()
        self.resource_inventory_set(rp1['uuid'], 'MEMORY_MB=1024', 'DISK_GB=256')
        self.resource_inventory_set(rp2['uuid'], 'MEMORY_MB=1024', 'DISK_GB=256')
        self.resource_inventory_set(rp3['uuid'], 'MEMORY_MB=1024', 'DISK_GB=256')
        self.resource_provider_trait_set(rp1['uuid'], 'STORAGE_DISK_SSD', 'HW_CPU_X86_BMI')
        self.resource_provider_trait_set(rp2['uuid'], 'STORAGE_DISK_HDD', 'HW_CPU_X86_BMI')
        self.resource_provider_trait_set(rp3['uuid'], 'STORAGE_DISK_HDD', 'HW_CPU_X86_BMI')
        rps = self.allocation_candidate_list(resources=('MEMORY_MB=1024', 'DISK_GB=80'), required=('HW_CPU_X86_BMI',), forbidden=('STORAGE_DISK_HDD',))
        self.assertEqual(1, len(rps))
        self.assertEqual(rp1['uuid'], rps[0]['resource provider'])
        rps = self.allocation_candidate_list(resources=('MEMORY_MB=1024', 'DISK_GB=80'), required=('HW_CPU_X86_BMI',), forbidden=('STORAGE_DISK_SSD',))
        uuids = [rp['resource provider'] for rp in rps]
        self.assertEqual(2, len(uuids))
        self.assertNotIn(rp1['uuid'], uuids)
        self.assertIn(rp2['uuid'], uuids)
        self.assertIn(rp3['uuid'], uuids)