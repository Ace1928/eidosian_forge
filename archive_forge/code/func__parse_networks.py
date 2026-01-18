from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def _parse_networks(net_list):
    network_cmd = []
    for net_dict in net_list:
        net_cmd = 'network '
        if net_dict.get('prefix'):
            net_cmd = net_cmd + net_dict.get('prefix')
        else:
            net_cmd = net_cmd + net_dict.get('network_address') + ' ' + net_dict.get('mask')
        if net_dict.get('area'):
            net_cmd = net_cmd + ' area ' + net_dict.get('area')
        network_cmd.append(net_cmd)
    return network_cmd