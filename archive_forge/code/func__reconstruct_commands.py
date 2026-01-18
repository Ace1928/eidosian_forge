from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def _reconstruct_commands(self, cmds):
    for idx, cmd in enumerate(cmds):
        match = re.search('^(?P<cmd>(no\\s)?switchport trunk allowed vlan(\\sadd)?)\\s(?P<vlans>.+)', cmd)
        if match:
            data = match.groupdict()
            unparsed = vlan_list_to_range(data['vlans'].split(','))
            cmds[idx] = data['cmd'] + ' ' + unparsed