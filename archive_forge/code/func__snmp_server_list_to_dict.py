from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.snmp_server import (
def _snmp_server_list_to_dict(self, entry):
    param_dict = {'communities': 'name', 'listen_addresses': 'address'}
    v3_param_dict = {'groups': 'group', 'users': 'user', 'views': 'view', 'trap_targets': 'address'}
    for k, v in iteritems(param_dict):
        if k in entry:
            a_dict = {}
            for el in entry[k]:
                a_dict.update({el[v]: el})
            entry[k] = a_dict
    for k, v in iteritems(v3_param_dict):
        if entry.get('snmp_v3') and k in entry.get('snmp_v3'):
            a_dict = {}
            for el in entry['snmp_v3'][k]:
                a_dict.update({el[v]: el})
            entry['snmp_v3'][k] = a_dict
    return entry