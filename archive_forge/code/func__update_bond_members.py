from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _update_bond_members(self, key, want, have):
    commands = []
    want_members = want.get(key) or []
    have_members = have.get(key) or []
    members_diff = list_diff_have_only(want_members, have_members)
    if members_diff:
        for member in members_diff:
            commands.append(self._compute_command(member['member'], 'bond-group', have['name'], True, 'ethernet'))
    return commands