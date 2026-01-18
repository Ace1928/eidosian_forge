from __future__ import (absolute_import, division, print_function)
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination
def _get_device_host(self, mgmt):
    if len(mgmt['DeviceManagement']) == 1 and mgmt['DeviceManagement'][0]['NetworkAddress'].startswith('['):
        dev_host = mgmt['DeviceManagement'][0]['NetworkAddress'][1:-1]
    elif len(mgmt['DeviceManagement']) == 2 and mgmt['DeviceManagement'][0]['NetworkAddress'].startswith('['):
        dev_host = mgmt['DeviceManagement'][1]['NetworkAddress']
    else:
        dev_host = mgmt['DeviceManagement'][0]['NetworkAddress']
    return dev_host