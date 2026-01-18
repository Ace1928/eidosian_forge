from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def _v6_cmds(self, want, have, state=''):
    """Helper method for processing ipv6 changes.
        This is needed to avoid unnecessary churn on the device when removing or changing multiple addresses.
        """
    for i in want:
        i['tag'] = i.get('tag')
    for i in have:
        i['tag'] = i.get('tag')
    cmds = []
    if state == 'replaced':
        for i in self.diff_list_of_dicts(have, want):
            want_addr = [w for w in want if w['address'] == i['address']]
            if not want_addr:
                cmds.append('no ipv6 address %s' % i['address'])
            elif i['tag'] and (not want_addr[0]['tag']):
                cmds.append('no ipv6 address %s' % i['address'])
    for i in self.diff_list_of_dicts(want, have):
        addr = i['address']
        tag = i['tag']
        if not tag and state == 'merged':
            have_addr = [h for h in have if h['address'] == addr]
            if have_addr and have_addr[0].get('tag'):
                continue
        cmd = 'ipv6 address %s' % i['address']
        cmd += ' tag %s' % tag if tag else ''
        cmds.append(cmd)
    return cmds