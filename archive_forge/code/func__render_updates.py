from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_updates(self, want, have):
    commands = []
    if have:
        temp_have_legacy_protos = have.pop('legacy_protocols', None)
    else:
        have = {}
        temp_want_legacy_protos = want.pop('legacy_protocols', None)
    updates = dict_diff(have, want)
    if have and temp_have_legacy_protos:
        have['legacy_protocols'] = temp_have_legacy_protos
    if not have and temp_want_legacy_protos:
        want['legacy_protocols'] = temp_want_legacy_protos
    commands.extend(self._add_lldp_protocols(want, have))
    if updates:
        for key, value in iteritems(updates):
            if value:
                if key == 'enable':
                    commands.append(self._compute_command())
                elif key == 'address':
                    commands.append(self._compute_command('management-address', str(value)))
                elif key == 'snmp':
                    if value == 'disable':
                        commands.append(self._compute_command(key, remove=True))
                    else:
                        commands.append(self._compute_command(key, str(value)))
    return commands