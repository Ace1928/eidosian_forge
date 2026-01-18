from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_export_domain_service(self):
    provider_name = self._module.params['export_domain']
    export_sds_service = self._connection.system_service().storage_domains_service()
    export_sd_id = get_id_by_name(export_sds_service, provider_name)
    return export_sds_service.service(export_sd_id)