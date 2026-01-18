from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _get_fc_interface(module, array):
    """Return FC Interface or None"""
    interface = {}
    interface_list = array.get_network_interfaces(names=[module.params['name']])
    if interface_list.status_code == 200:
        interface = list(interface_list.items)[0]
        return interface
    return None