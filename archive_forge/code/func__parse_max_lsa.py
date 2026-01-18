from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def _parse_max_lsa(max_lsa_dict):
    max_lsa_cmd = 'max-lsa '
    if max_lsa_dict.get('count'):
        max_lsa_cmd = max_lsa_cmd + ' ' + str(max_lsa_dict['count'])
    if max_lsa_dict.get('threshold'):
        max_lsa_cmd = max_lsa_cmd + ' ' + str(max_lsa_dict['threshold'])
    for lsa_key, lsa_val in sorted(max_lsa_dict.items()):
        if lsa_key == 'warning' and lsa_val:
            max_lsa_cmd = max_lsa_cmd + ' warning-only'
        elif lsa_key in ['ignore_count', 'reset_time', 'ignore_time']:
            if lsa_val:
                k = re.sub('_', '-', lsa_key)
                max_lsa_cmd = max_lsa_cmd + ' ' + k + ' ' + str(lsa_val) + ' '
    return max_lsa_cmd