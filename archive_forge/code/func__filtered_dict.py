from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.vxlan_vtep import (
def _filtered_dict(self, want, have):
    """Remove other config from 'have' if 'member' key is present"""
    if 'member' in want:
        have_member = {}
        want_vni_dict = want.get('member', {}).get('vni', {})
        have_vni_dict = have.get('member', {}).get('vni', {})
        for vni_type, have_vnis in have_vni_dict.items():
            want_vnis = want_vni_dict.get(vni_type, {})
            have_member[vni_type] = {vni: have_vni_dict[vni_type].get(vni) for vni in have_vnis if vni in want_vnis}
        have = {'interface': have['interface'], 'member': {'vni': have_member}}
    return have