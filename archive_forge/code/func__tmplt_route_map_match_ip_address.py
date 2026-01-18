from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_map_match_ip_address(config_data):
    command = ''
    config_data = config_data['entries']['match']['ip']
    if config_data.get('address'):
        config_data = config_data['address']
        command = 'match ip address '
        if config_data.get('dynamic'):
            command += 'dynamic'
        if config_data.get('access_list'):
            command += 'access-list ' + config_data['access_list']
        if config_data.get('prefix_list'):
            command += 'prefix-list ' + config_data['prefix_list']
    return command