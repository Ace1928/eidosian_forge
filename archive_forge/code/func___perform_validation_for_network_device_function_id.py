from __future__ import absolute_import, division, print_function
import json
import time
from urllib.error import HTTPError, URLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import (
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def __perform_validation_for_network_device_function_id(self):
    odata = '@odata.id'
    network_device_function_id_uri, found_device = ('', False)
    network_device_function_id = self.module.params.get('network_device_function_id')
    network_adapter_id_uri = self.__perform_validation_for_network_adapter_id()
    network_devices = get_dynamic_uri(self.idrac, network_adapter_id_uri, 'NetworkDeviceFunctions')[odata]
    network_device_list = get_dynamic_uri(self.idrac, network_devices, 'Members')
    for each_device in network_device_list:
        if network_device_function_id in each_device.get(odata):
            found_device = True
            network_device_function_id_uri = each_device.get(odata)
            break
    if not found_device:
        self.module.exit_json(failed=True, msg=INVALID_ID_MSG.format(network_device_function_id, 'network_device_function_id'))
    return network_device_function_id_uri