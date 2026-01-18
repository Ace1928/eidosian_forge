from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_qos_id(self):
    if self.param('qos'):
        qoss_service = self._get_dcs_service().service(self._get_dcs_id()).qoss_service()
        return get_id_by_name(qoss_service, self.param('qos')) if self.param('qos') else None
    return None