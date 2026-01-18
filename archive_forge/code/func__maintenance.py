from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _maintenance(self, storage_domain):
    attached_sd_service = self._attached_sd_service(storage_domain)
    attached_sd = get_entity(attached_sd_service)
    if attached_sd and attached_sd.status != sdstate.MAINTENANCE:
        if not self._module.check_mode:
            attached_sd_service.deactivate()
        self.changed = True
        wait(service=attached_sd_service, condition=lambda sd: sd.status == sdstate.MAINTENANCE, wait=self.param('wait'), timeout=self.param('timeout'))