from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_af_neighbor_prefix_list(config_data):
    command = []
    afi = config_data['neighbors']['address_family']['afi'] + '-unicast'
    cmd = 'protocols bgp {as_number} neighbor '.format(**config_data)
    cmd += '{neighbor_address} address-family '.format(**config_data['neighbors'])
    config_data = config_data['neighbors']['address_family']
    for list_el in config_data['prefix_list']:
        command.append(cmd + afi + ' prefix-list ' + list_el['action'] + ' ' + str(list_el['prefix_list']))
    return command