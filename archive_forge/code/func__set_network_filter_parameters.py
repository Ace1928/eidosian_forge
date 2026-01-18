from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _set_network_filter_parameters(self, entity_id):
    if self._module.params['network_filter_parameters'] is not None:
        nfps_service = self._service.service(entity_id).network_filter_parameters_service()
        nfp_list = nfps_service.list()
        for nfp in nfp_list:
            nfps_service.service(nfp.id).remove()
        for nfp in self._network_filter_parameters():
            nfps_service.add(nfp)