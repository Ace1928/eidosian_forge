import uuid
from osc_placement.tests.functional import base
class TestAllocationCandidate117(base.BaseTestCase):
    VERSION = '1.17'

    def test_show_required_trait(self):
        rp1 = self.resource_provider_create()
        rp2 = self.resource_provider_create()
        self.resource_inventory_set(rp1['uuid'], 'MEMORY_MB=8192', 'DISK_GB=512')
        self.resource_inventory_set(rp2['uuid'], 'MEMORY_MB=8192', 'DISK_GB=512')
        self.resource_provider_trait_set(rp1['uuid'], 'STORAGE_DISK_SSD', 'HW_NIC_SRIOV')
        self.resource_provider_trait_set(rp2['uuid'], 'STORAGE_DISK_HDD', 'HW_NIC_SRIOV')
        rps = self.allocation_candidate_list(resources=('MEMORY_MB=1024', 'DISK_GB=80'), required=('STORAGE_DISK_SSD',))
        candidate_dict = {rp['resource provider']: rp for rp in rps}
        self.assertIn(rp1['uuid'], candidate_dict)
        self.assertNotIn(rp2['uuid'], candidate_dict)
        self.assertEqual(set(candidate_dict[rp1['uuid']]['traits'].split(',')), set(['STORAGE_DISK_SSD', 'HW_NIC_SRIOV']))

    def test_fail_if_aggregate_uuid_wrong_version(self):
        self.assertCommandFailed('Operation or argument is not supported with version 1.17', self.allocation_candidate_list, resources=('MEMORY_MB=1024', 'DISK_GB=80'), aggregate_uuids=[str(uuid.uuid4())])
        self.assertCommandFailed('Operation or argument is not supported with version 1.17', self.allocation_candidate_list, resources=('MEMORY_MB=1024', 'DISK_GB=80'), member_of=[str(uuid.uuid4())])