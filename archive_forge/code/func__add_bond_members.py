from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _add_bond_members(self, want, have):
    commands = []
    diff_members = get_lst_diff_for_dicts(want, have, 'members')
    if diff_members:
        for key in diff_members:
            commands.append(self._compute_command(key['member'], 'bond-group', want['name'], type='ethernet'))
    return commands