from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def _parse_summary_address(addr_dict):
    sum_cmd = 'summary-address '
    if addr_dict.get('prefix'):
        sum_cmd = sum_cmd + addr_dict.get('prefix')
    else:
        sum_cmd = sum_cmd + addr_dict.get('address') + ' ' + addr_dict.get('mask')
    if 'attribute_map' in addr_dict.keys():
        sum_cmd = sum_cmd + ' attribute-map ' + addr_dict['attribute_map']
    elif addr_dict.get('not_advertise'):
        sum_cmd = sum_cmd + ' not-advertise '
    elif 'tag' in addr_dict.keys():
        sum_cmd = sum_cmd + ' tag ' + addr_dict['tag']
    return sum_cmd