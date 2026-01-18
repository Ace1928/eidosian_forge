import collections
import copy
import uuid
from osc_placement.tests.functional import base
def _get_expected_inventories(self, old_inventories, resources):
    new_inventories = []
    for old_inventory in old_inventories:
        new_inventory = collections.defaultdict(dict)
        new_inventory.update(copy.deepcopy(old_inventory))
        for resource in resources:
            rc, keyval = resource.split(':')
            key, val = keyval.split('=')
            val = float(val) if '.' in val else int(val)
            new_inventory[rc][key] = val
            if 'resource_class' not in new_inventory[rc]:
                new_inventory[rc]['resource_class'] = rc
        new_inventories.append(new_inventory)
    return new_inventories