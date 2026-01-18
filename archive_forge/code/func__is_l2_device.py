from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.vlans.vlans import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.config.vlans.vlans import Vlans
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import get_connection
def _is_l2_device(module):
    """fails module if device is L3."""
    connection = get_connection(module)
    check_os_type = connection.get_device_info()
    if check_os_type.get('network_os_type') == 'L3':
        return False
    return True