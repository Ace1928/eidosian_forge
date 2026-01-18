from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
from ansible_collections.arista.eos.plugins.module_utils.network.eos.rm_templates.prefix_lists import (
def _compare_seq(self, afi, w, h):
    wl_child = {}
    hl_child = {}
    parser = ['prefixlist.entry', 'prefixlist.resequence']
    for seq, ent in iteritems(w):
        seq_diff = {}
        wl_child = {'afi': afi, 'prefix_lists': {'entries': {seq: ent}}}
        if h.get(seq):
            hl_child = {'afi': afi, 'prefix_lists': {'entries': {seq: h.pop(seq)}}}
            seq_diff = dict_diff(hl_child['prefix_lists']['entries'][seq], wl_child['prefix_lists']['entries'][seq])
        if seq_diff:
            if self.state == 'merged':
                self._module.fail_json(msg='Sequence number ' + str(seq) + ' is already present. Use replaced/overridden operation to change the configuration')
            self.compare(parsers='prefixlist.entry', want={}, have=hl_child)
        self.compare(parsers=parser, want=wl_child, have=hl_child)