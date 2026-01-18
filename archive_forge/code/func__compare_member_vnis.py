from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.vxlan_vtep import (
def _compare_member_vnis(self, wantmv, havemv):
    """Compare member VNIs dict"""
    PARSER_DICT = {'l2vni': 'replication', 'l3vni': 'vrf'}
    for vni_type in ['l2vni', 'l3vni']:
        wantd = wantmv.get(vni_type, {})
        haved = havemv.get(vni_type, {})
        undel_vnis = haved.copy()
        for wvni, want in wantd.items():
            have = haved.pop(wvni, {})
            if want != have:
                self.addcmd(undel_vnis.pop(wvni, {}), PARSER_DICT[vni_type], True)
                if vni_type == 'l3vni':
                    undel_vnis = self._remove_existing_vnis_vrfs(want['vrf'], undel_vnis)
                self.addcmd(want, PARSER_DICT[vni_type])
        for hvni, have in haved.items():
            if hvni in undel_vnis:
                self.addcmd(have, PARSER_DICT[vni_type], True)