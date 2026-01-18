import collections
import copy
import uuid
from osc_placement.tests.functional import base
class TestInventory15(TestInventory):
    VERSION = '1.5'

    def test_delete_all_inventories(self):
        rp = self.resource_provider_create()
        self.resource_inventory_set(rp['uuid'], 'MEMORY_MB=16', 'VCPU=32')
        self.resource_inventory_delete(rp['uuid'])
        self.assertEqual([], self.resource_inventory_list(rp['uuid']))