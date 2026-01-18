import uuid
from osc_placement.tests.functional import base
class TestAllocationCandidate121(base.BaseTestCase):
    VERSION = '1.21'

    def test_return_properly_for_aggregate_uuid_request(self):
        rp1 = self.resource_provider_create()
        rp2 = self.resource_provider_create()
        self.resource_inventory_set(rp1['uuid'], 'MEMORY_MB=8192', 'DISK_GB=512')
        self.resource_inventory_set(rp2['uuid'], 'MEMORY_MB=8192', 'DISK_GB=512')
        agg = str(uuid.uuid4())
        self.resource_provider_aggregate_set(rp2['uuid'], agg, generation=1)
        rps, warning = self.allocation_candidate_list(resources=('MEMORY_MB=1024',), aggregate_uuids=[agg, str(uuid.uuid4())], may_print_to_stderr=True)
        candidate_dict = {rp['resource provider']: rp for rp in rps}
        self.assertEqual(1, len(candidate_dict))
        self.assertIn(rp2['uuid'], candidate_dict)
        self.assertNotIn(rp1['uuid'], candidate_dict)
        self.assertIn('The --aggregate-uuid option is deprecated, please use --member-of instead.', warning)
        rps = self.allocation_candidate_list(resources=('MEMORY_MB=1024',), member_of=[agg])
        candidate_dict = {rp['resource provider']: rp for rp in rps}
        self.assertEqual(1, len(candidate_dict))
        self.assertIn(rp2['uuid'], candidate_dict)
        self.assertNotIn(rp1['uuid'], candidate_dict)

    def test_fail_if_forbidden_trait_wrong_version(self):
        self.assertCommandFailed('Operation or argument is not supported with version 1.21', self.allocation_candidate_list, resources=('MEMORY_MB=1024', 'DISK_GB=80'), forbidden=('STORAGE_DISK_HDD',))