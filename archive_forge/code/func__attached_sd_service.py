from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _attached_sd_service(self, storage_domain):
    dc_name = self.param('data_center')
    if not dc_name:
        dc_name = self._find_attached_datacenter_name(storage_domain.name)
    attached_sds_service = self._attached_sds_service(dc_name)
    attached_sd_service = attached_sds_service.storage_domain_service(storage_domain.id)
    return attached_sd_service