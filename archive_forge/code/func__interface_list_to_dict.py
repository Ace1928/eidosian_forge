from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.vxlan_vtep import (
def _interface_list_to_dict(self, want, have):
    """Convert all list of dicts to dicts of dicts"""
    wantd = {entry['interface']: entry for entry in want}
    haved = {entry['interface']: entry for entry in have}
    for each in (wantd, haved):
        if each:
            for nvi, nvid in each.items():
                member_vni = nvid.get('member', {}).get('vni')
                if member_vni:
                    for vni_type in member_vni:
                        member_vni[vni_type] = param_list_to_dict(member_vni[vni_type], unique_key='vni', remove_key=False)
    return (wantd, haved)