import uuid
from osc_placement.tests.functional import base
class TestAllocationCandidate116(base.BaseTestCase):
    VERSION = '1.16'

    def test_list_limit(self):
        rp1 = self.resource_provider_create()
        rp2 = self.resource_provider_create()
        self.resource_inventory_set(rp1['uuid'], 'MEMORY_MB=8192', 'DISK_GB=512')
        self.resource_inventory_set(rp2['uuid'], 'MEMORY_MB=8192', 'DISK_GB=512')
        unlimited = self.allocation_candidate_list(resources=('MEMORY_MB=1024', 'DISK_GB=80'))
        self.assertTrue(len(set([row['#'] for row in unlimited])) > 1)
        limited = self.allocation_candidate_list(resources=('MEMORY_MB=1024', 'DISK_GB=80'), limit=1)
        self.assertEqual(len(set([row['#'] for row in limited])), 1)