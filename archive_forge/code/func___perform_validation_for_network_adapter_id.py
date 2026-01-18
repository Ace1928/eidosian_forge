from __future__ import absolute_import, division, print_function
import json
import time
from urllib.error import HTTPError, URLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import (
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def __perform_validation_for_network_adapter_id(self):
    odata = '@odata.id'
    network_adapter_id = self.module.params.get('network_adapter_id')
    network_adapter_id_uri, found_adapter = ('', False)
    uri, error_msg = validate_and_get_first_resource_id_uri(self.module, self.idrac, SYSTEMS_URI)
    if error_msg:
        self.module.exit_json(msg=error_msg, failed=True)
    network_adapters = get_dynamic_uri(self.idrac, uri, 'NetworkInterfaces')[odata]
    network_adapter_list = get_dynamic_uri(self.idrac, network_adapters, 'Members')
    for each_adapter in network_adapter_list:
        if network_adapter_id in each_adapter.get(odata):
            found_adapter = True
            network_adapter_id_uri = each_adapter.get(odata)
            break
    if not found_adapter:
        self.module.exit_json(failed=True, msg=INVALID_ID_MSG.format(network_adapter_id, 'network_adapter_id'))
    return network_adapter_id_uri