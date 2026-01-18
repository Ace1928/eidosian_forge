import collections
import copy
import uuid
from osc_placement.tests.functional import base
def _test_dry_run(self, agg, rps, old_inventories, amend=False):
    new_resources = ['VCPU:allocation_ratio=5.0', 'MEMORY_MB:allocation_ratio=6.0', 'DISK_GB:allocation_ratio=7.0']
    resp = self.resource_inventory_set(agg, *new_resources, aggregate=True, amend=amend, dry_run=True)
    inventories = old_inventories if amend else [{}] * len(old_inventories)
    new_inventories = self._get_expected_inventories(inventories, new_resources)
    expected = {}
    for rp, inventory in zip(rps, new_inventories):
        for rc, inv in inventory.items():
            inv['resource_provider'] = rp['uuid']
            for key in ('max_unit', 'min_unit', 'reserved', 'step_size', 'total', 'reserved', 'step_size'):
                if key not in inv:
                    inv[key] = ''
        expected[rp['uuid']] = inventory
    resp_dict = collections.defaultdict(dict)
    for row in resp:
        resp_dict[row['resource_provider']][row['resource_class']] = row
    self.assertEqual(expected, resp_dict)
    for i, rp in enumerate(rps):
        resp = self.resource_inventory_list(rp['uuid'])
        self.assertDictEqual(old_inventories[i], {r['resource_class']: r for r in resp})