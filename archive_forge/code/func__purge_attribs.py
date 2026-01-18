from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _purge_attribs(self, have):
    commands = []
    for item in Lag_interfaces.params:
        if have.get(item):
            if item == 'members':
                commands.extend(self._delete_bond_members(have))
            elif item != 'name':
                commands.append(self._compute_command(have['name'], attrib=item, remove=True))
    return commands