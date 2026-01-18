from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def _parse_areas_filter_notsostubby(nss_dict):
    nss_cmd = 'not-so-stubby '
    if nss_dict.get('default_information_originate'):
        nss_cmd = nss_cmd + 'default-information-originate '
        for def_keys in nss_dict['default_information_originate'].keys():
            if def_keys == 'nssa_only' and nss_dict['default_information_originate']['nssa_only']:
                nss_cmd = nss_cmd + ' nssa-only '
            elif nss_dict['default_information_originate'].get(def_keys):
                nss_cmd = nss_cmd + def_keys + ' ' + nss_dict['default_information_originate'][def_keys]
    elif 'lsa' in nss_dict.keys() and nss_dict.get('lsa'):
        nss_cmd = nss_cmd + ' lsa type-7 convert type-5'
    elif 'no_summary' in nss_dict.keys() and nss_dict.get('no_summary'):
        nss_cmd = nss_cmd + ' no-summary'
    elif 'nssa_only' in nss_dict.keys() and nss_dict.get('nssa_only'):
        nss_cmd = nss_cmd + ' nssa-only'
    return nss_cmd