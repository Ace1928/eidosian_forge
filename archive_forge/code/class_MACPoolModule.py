from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class MACPoolModule(BaseModule):

    def build_entity(self):
        return otypes.MacPool(name=self._module.params['name'], id=self._module.params['id'], allow_duplicates=self._module.params['allow_duplicates'], description=self._module.params['description'], ranges=[otypes.Range(from_=mac_range.split(',')[0], to=mac_range.split(',')[1]) for mac_range in self._module.params['ranges']] if self._module.params['ranges'] else None)

    def _compare_ranges(self, entity):
        if self._module.params['ranges'] is not None:
            ranges = sorted(['%s,%s' % (mac_range.from_, mac_range.to) for mac_range in entity.ranges])
            return equal(sorted(self._module.params['ranges']), ranges)
        return True

    def update_check(self, entity):
        return self._compare_ranges(entity) and equal(self._module.params['allow_duplicates'], entity.allow_duplicates) and equal(self._module.params['description'], entity.description) and equal(self._module.params['name'], entity.name)