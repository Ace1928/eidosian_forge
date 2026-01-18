from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.vxlan_vtep import (
def _remove_existing_vnis_vrfs(self, want_vrf, haved):
    """Remove member VNIs of corresponding VRF"""
    vrf_haved = next((h for h in haved.values() if h['vrf'] == want_vrf), None)
    if vrf_haved:
        self.addcmd(haved.pop(vrf_haved['vni']), 'vrf', True)
    return haved