from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_del_commands(self, want, have):
    commands = []
    params = Lag_interfaces.params
    for attrib in params:
        if attrib == 'members':
            commands.extend(self._update_bond_members(attrib, want, have))
        elif attrib == 'arp_monitor':
            commands.extend(self._update_arp_monitor(attrib, want, have))
        elif have.get(attrib) and (not want.get(attrib)):
            commands.append(self._compute_command(have['name'], attrib, remove=True))
    return commands