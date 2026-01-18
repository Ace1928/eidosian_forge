from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def __derive_system_config_delete_op(key_set, command, exist_conf):
    new_conf = exist_conf
    if 'hostname' in command:
        new_conf['hostname'] = 'sonic'
    if 'interface_naming' in command:
        new_conf['interface_naming'] = 'native'
    if 'anycast_address' in command and 'anycast_address' in new_conf:
        if 'ipv4' in command['anycast_address']:
            new_conf['anycast_address']['ipv4'] = True
        if 'ipv6' in command['anycast_address']:
            new_conf['anycast_address']['ipv6'] = True
        if 'mac_address' in command['anycast_address']:
            new_conf['anycast_address']['mac_address'] = None
    return (True, new_conf)