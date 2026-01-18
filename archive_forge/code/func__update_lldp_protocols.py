from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _update_lldp_protocols(self, want_item, have_item):
    commands = []
    want_protocols = want_item.get('legacy_protocols') or []
    have_protocols = have_item.get('legacy_protocols') or []
    members_diff = list_diff_have_only(want_protocols, have_protocols)
    if members_diff:
        for member in members_diff:
            commands.append(self._compute_command('legacy-protocols', member, remove=True))
    return commands