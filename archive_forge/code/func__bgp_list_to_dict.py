from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.bgp_neighbor_address_family import (
def _bgp_list_to_dict(self, data):
    if 'neighbors' in data:
        for nbr in data['neighbors']:
            if 'address_family' in nbr:
                nbr['address_family'] = {(x['afi'], x.get('safi')): x for x in nbr['address_family']}
        data['neighbors'] = {x['neighbor_address']: x for x in data['neighbors']}
    if 'vrfs' in data:
        for vrf in data['vrfs']:
            self._bgp_list_to_dict(vrf)
        data['vrfs'] = {x['vrf']: x for x in data['vrfs']}