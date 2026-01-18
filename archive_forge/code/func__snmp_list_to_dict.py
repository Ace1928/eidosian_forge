from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.snmp_server import (
def _snmp_list_to_dict(self, data):
    """Convert all list of dicts to dicts of dicts"""
    p_key = {'hosts': 'host', 'groups': 'group', 'engine_id': 'id', 'communities': 'name', 'context': True, 'password_policy': 'policy_name', 'file_transfer': True, 'users': 'username', 'views': 'name'}
    tmp_data = deepcopy(data)
    for k, _v in p_key.items():
        if k in tmp_data:
            if k == 'hosts':
                tmp_host = dict()
                for i in tmp_data[k]:
                    tmp = dict()
                    if i.get('traps'):
                        for t in i.get('traps'):
                            tmp.update({t: t})
                        i['traps'] = tmp
                    tmp_host.update({str(i[p_key.get(k)] + i.get('version', '') + i.get('community_string', '')): i})
                tmp_data[k] = tmp_host
            elif k == 'context':
                tmp_data[k] = {i: {'context': i} for i in tmp_data[k]}
            elif k == 'file_transfer':
                if tmp_data.get(k):
                    if tmp_data[k].get('protocol'):
                        tmp = dict()
                        for t in tmp_data[k].get('protocol'):
                            tmp.update({t: t})
                        tmp_data[k]['protocol'] = tmp
            elif k == 'groups':
                tmp_data[k] = {str(i[p_key.get(k)] + i.get('version_option', '') + i.get('context', '')): i for i in tmp_data[k]}
            elif k == 'views':
                tmp_data[k] = {str(i[p_key.get(k)] + i.get('family_name', '')): i for i in tmp_data[k]}
            else:
                tmp_data[k] = {str(i[p_key.get(k)]): i for i in tmp_data[k]}
    return tmp_data