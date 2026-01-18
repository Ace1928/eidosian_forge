from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_map_extcommunity_soo(config_data):
    config_data = config_data['entries']['set']['extcommunity']['soo']
    command = 'set extcommunity soo ' + config_data['vpn']
    if config_data.get('additive'):
        command += ' additive'
    if config_data.get('delete'):
        command += ' delete'
    return command