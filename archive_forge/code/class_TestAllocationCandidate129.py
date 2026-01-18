import uuid
from osc_placement.tests.functional import base
class TestAllocationCandidate129(base.BaseTestCase):
    VERSION = '1.29'

    def setUp(self):
        super(TestAllocationCandidate129, self).setUp()
        self.rp1 = self.resource_provider_create()
        self.rp1_1 = self.resource_provider_create(parent_provider_uuid=self.rp1['uuid'])
        self.rp1_2 = self.resource_provider_create(parent_provider_uuid=self.rp1['uuid'])
        self.agg1 = str(uuid.uuid4())
        self.agg2 = str(uuid.uuid4())
        self.resource_provider_aggregate_set(self.rp1_1['uuid'], self.agg1, generation=0)
        self.resource_provider_aggregate_set(self.rp1_2['uuid'], self.agg2, generation=0)
        self.resource_inventory_set(self.rp1['uuid'], 'DISK_GB=512')
        self.resource_inventory_set(self.rp1_1['uuid'], 'VCPU=8', 'MEMORY_MB=8192')
        self.resource_inventory_set(self.rp1_2['uuid'], 'VCPU=16', 'MEMORY_MB=8192')
        self.resource_provider_trait_set(self.rp1_1['uuid'], 'HW_CPU_X86_AVX')
        self.resource_provider_trait_set(self.rp1_2['uuid'], 'HW_CPU_X86_SSE')

    def test_granular_one_group(self):
        groups = {'1': {'resources': ('VCPU=3',)}}
        rows = self.allocation_candidate_granular(groups=groups)
        self.assertEqual(2, len(rows))
        numbers = {row['#'] for row in rows}
        self.assertEqual(2, len(numbers))
        rps = {row['resource provider'] for row in rows}
        self.assertEqual(2, len(rps))
        self.assertIn(self.rp1_1['uuid'], rps)
        self.assertIn(self.rp1_2['uuid'], rps)

    def test_granular_two_groups(self):
        groups = {'1': {'resources': ('VCPU=3',)}, '2': {'resources': ('VCPU=3',)}}
        rows = self.allocation_candidate_granular(groups=groups)
        self.assertEqual(6, len(rows))
        numbers = {row['#'] for row in rows}
        self.assertEqual(4, len(numbers))
        rps = {row['resource provider'] for row in rows}
        self.assertEqual(2, len(rps))
        self.assertIn(self.rp1_1['uuid'], rps)
        self.assertIn(self.rp1_2['uuid'], rps)
        rows = self.allocation_candidate_granular(groups=groups, group_policy='isolate')
        self.assertEqual(4, len(rows))
        numbers = {row['#'] for row in rows}
        self.assertEqual(2, len(numbers))
        rps = {row['resource provider'] for row in rows}
        self.assertEqual(2, len(rps))
        self.assertIn(self.rp1_1['uuid'], rps)
        self.assertIn(self.rp1_2['uuid'], rps)
        rows = self.allocation_candidate_granular(groups=groups, group_policy='isolate', limit=1)
        self.assertEqual(2, len(rows))
        numbers = {row['#'] for row in rows}
        self.assertEqual(1, len(numbers))
        rps = {row['resource provider'] for row in rows}
        self.assertEqual(2, len(rps))
        self.assertIn(self.rp1_1['uuid'], rps)
        self.assertIn(self.rp1_2['uuid'], rps)

    def test_granular_traits1(self):
        groups = {'1': {'resources': ('VCPU=6',)}, '2': {'resources': ('VCPU=10',), 'required': ['HW_CPU_X86_AVX']}}
        rows = self.allocation_candidate_granular(groups=groups, group_policy='isolate')
        self.assertEqual(0, len(rows))

    def test_granular_traits2(self):
        groups = {'1': {'resources': ('VCPU=6',)}, '2': {'resources': ('VCPU=10',), 'required': ['HW_CPU_X86_SSE']}}
        rows = self.allocation_candidate_granular(groups=groups, group_policy='isolate')
        self.assertEqual(2, len(rows))
        numbers = {row['#'] for row in rows}
        self.assertEqual(1, len(numbers))
        rps = {row['resource provider'] for row in rows}
        self.assertEqual(2, len(rps))
        self.assertIn(self.rp1_1['uuid'], rps)
        self.assertIn(self.rp1_2['uuid'], rps)

    def test_list_with_any_traits_old_microversion(self):
        groups = {'': {'resources': ('DISK_GB=1',), 'required': ('STORAGE_DISK_HDD,STORAGE_DISK_SSD',)}, '1': {'resources': ('VCPU=1',), 'required': ('HW_CPU_X86_AVX',)}}
        self.assertCommandFailed('Operation or argument is not supported with version 1.29', self.allocation_candidate_granular, groups=groups)