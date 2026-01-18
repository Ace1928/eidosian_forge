import uuid
from osc_placement.tests.functional import base
class TestAllocation112(base.BaseTestCase):
    VERSION = '1.12'

    def setUp(self):
        super(TestAllocation112, self).setUp()
        self.rp1 = self.resource_provider_create()
        self.resource_inventory_set(self.rp1['uuid'], 'VCPU=4', 'MEMORY_MB=1024')

    def test_allocation_update(self):
        consumer_uuid = str(uuid.uuid4())
        project_uuid = str(uuid.uuid4())
        user_uuid = str(uuid.uuid4())
        created_alloc = self.resource_allocation_set(consumer_uuid, ['rp={},VCPU=2'.format(self.rp1['uuid']), 'rp={},MEMORY_MB=512'.format(self.rp1['uuid'])], project_id=project_uuid, user_id=user_uuid)
        retrieved_alloc = self.resource_allocation_show(consumer_uuid)
        expected = [{'resource_provider': self.rp1['uuid'], 'generation': 2, 'project_id': project_uuid, 'user_id': user_uuid, 'resources': {'VCPU': 2, 'MEMORY_MB': 512}}]
        self.assertEqual(expected, created_alloc)
        self.assertEqual(expected, retrieved_alloc)

    def test_allocation_update_to_empty(self):
        consumer_uuid = str(uuid.uuid4())
        project_uuid = str(uuid.uuid4())
        user_uuid = str(uuid.uuid4())
        self.resource_allocation_set(consumer_uuid, ['rp={},VCPU=2'.format(self.rp1['uuid'])], project_id=project_uuid, user_id=user_uuid)
        result = self.resource_allocation_unset(consumer_uuid, columns=('resources',))
        self.assertEqual([], result)

    def test_allocation_show_empty(self):
        alloc = self.resource_allocation_show(str(uuid.uuid4()), columns=('resources',))
        self.assertEqual([], alloc)