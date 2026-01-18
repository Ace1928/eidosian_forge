from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_map_ip(config_data):
    config_data = config_data['entries']['set']
    if config_data.get('ip'):
        command = 'set ip next-hop '
        k = 'ip'
    elif config_data.get('ipv6'):
        command = 'set ip next-hop '
        k = 'ipv6'
    if config_data[k].get('address'):
        command += config_data[k]['address']
    elif config_data[k].get('unchanged'):
        command += 'unchanged'
    elif config_data[k].get('peer_address'):
        command += 'peer-address'
    return command