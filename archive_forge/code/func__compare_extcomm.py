from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.route_maps import (
def _compare_extcomm(self, want, have):
    hentry = get_from_dict(data_dict=have, keypath='set.extcommunity.rt') or {}
    wentry = get_from_dict(data_dict=want, keypath='set.extcommunity.rt') or {}
    h_nums = set(hentry.get('extcommunity_numbers', []))
    w_nums = set(wentry.get('extcommunity_numbers', []))
    if h_nums != w_nums or wentry.get('additive') != hentry.get('additive'):
        if self.state not in ['merged', 'rendered']:
            self.commands.append(self._tmplt.render(hentry, 'set.extcommunity.rt', negate=True))
        self.commands.append(self._tmplt.render(wentry, 'set.extcommunity.rt', negate=False))