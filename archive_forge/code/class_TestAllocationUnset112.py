import uuid
from osc_placement.tests.functional import base
class TestAllocationUnset112(base.BaseTestCase):
    VERSION = '1.12'

    def setUp(self):
        super(TestAllocationUnset112, self).setUp()
        self.rp1 = self.resource_provider_create()
        self.rp2 = self.resource_provider_create()
        self.rp3 = self.resource_provider_create()
        self.rp4 = self.resource_provider_create()
        self.resource_inventory_set(self.rp1['uuid'], 'VCPU=4', 'MEMORY_MB=1024')
        self.resource_inventory_set(self.rp2['uuid'], 'VGPU=1')
        self.resource_inventory_set(self.rp3['uuid'], 'VCPU=4', 'MEMORY_MB=1024', 'VGPU=1')
        self.resource_inventory_set(self.rp4['uuid'], 'VCPU=4', 'MEMORY_MB=1024', 'VGPU=1')
        self.consumer_uuid1 = str(uuid.uuid4())
        self.consumer_uuid2 = str(uuid.uuid4())
        self.project_uuid = str(uuid.uuid4())
        self.user_uuid = str(uuid.uuid4())
        self.resource_allocation_set(self.consumer_uuid1, ['rp={},VCPU=2'.format(self.rp1['uuid']), 'rp={},MEMORY_MB=512'.format(self.rp1['uuid']), 'rp={},VGPU=1'.format(self.rp2['uuid'])], project_id=self.project_uuid, user_id=self.user_uuid)
        self.resource_allocation_set(self.consumer_uuid2, ['rp={},VCPU=1'.format(self.rp3['uuid']), 'rp={},MEMORY_MB=256'.format(self.rp3['uuid']), 'rp={},VGPU=1'.format(self.rp3['uuid']), 'rp={},VCPU=1'.format(self.rp4['uuid']), 'rp={},MEMORY_MB=256'.format(self.rp4['uuid'])], project_id=self.project_uuid, user_id=self.user_uuid)

    def test_allocation_unset_one_provider(self):
        """Tests removing allocations for one specific provider."""
        updated_allocs = self.resource_allocation_unset(self.consumer_uuid1, provider=self.rp1['uuid'])
        expected = [{'resource_provider': self.rp2['uuid'], 'generation': 3, 'project_id': self.project_uuid, 'user_id': self.user_uuid, 'resources': {'VGPU': 1}}]
        self.assertEqual(expected, updated_allocs)

    def test_allocation_unset_one_resource_class(self):
        """Tests removing allocations for resource classes."""
        updated_allocs = self.resource_allocation_unset(self.consumer_uuid2, resource_class=['MEMORY_MB'])
        expected = [{'resource_provider': self.rp3['uuid'], 'generation': 3, 'project_id': self.project_uuid, 'user_id': self.user_uuid, 'resources': {'VCPU': 1, 'VGPU': 1}}, {'resource_provider': self.rp4['uuid'], 'generation': 3, 'project_id': self.project_uuid, 'user_id': self.user_uuid, 'resources': {'VCPU': 1}}]
        self.assertEqual(expected, updated_allocs)

    def test_allocation_unset_resource_classes(self):
        """Tests removing allocations for resource classes."""
        updated_allocs = self.resource_allocation_unset(self.consumer_uuid2, resource_class=['VCPU', 'MEMORY_MB'])
        expected = [{'resource_provider': self.rp3['uuid'], 'generation': 3, 'project_id': self.project_uuid, 'user_id': self.user_uuid, 'resources': {'VGPU': 1}}]
        self.assertEqual(expected, updated_allocs)

    def test_allocation_unset_provider_and_rc(self):
        """Tests removing allocations of resource classes for a provider ."""
        updated_allocs = self.resource_allocation_unset(self.consumer_uuid2, provider=self.rp3['uuid'], resource_class=['VCPU', 'MEMORY_MB'])
        expected = [{'resource_provider': self.rp3['uuid'], 'generation': 3, 'project_id': self.project_uuid, 'user_id': self.user_uuid, 'resources': {'VGPU': 1}}, {'resource_provider': self.rp4['uuid'], 'generation': 3, 'project_id': self.project_uuid, 'user_id': self.user_uuid, 'resources': {'VCPU': 1, 'MEMORY_MB': 256}}]
        self.assertEqual(expected, updated_allocs)

    def test_allocation_unset_remove_all_providers(self):
        """Tests removing all allocations by omitting the --provider option."""
        updated_allocs = self.resource_allocation_unset(self.consumer_uuid1, use_json=False)
        self.assertEqual('', updated_allocs.strip())