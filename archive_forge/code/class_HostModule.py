from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class HostModule(BaseModule):

    def build_entity(self):
        return otypes.Host(power_management=otypes.PowerManagement(enabled=True))

    def update_check(self, entity):
        return equal(True, entity.power_management.enabled)