from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.bgp_neighbor_address_family import (
def _set_to_delete(self, haved, wantd):
    neighbors = {}
    h_nbrs = haved.get('neighbors', {})
    w_nbrs = wantd.get('neighbors', {})
    for k, h_nbr in iteritems(h_nbrs):
        w_nbr = w_nbrs.pop(k, {})
        if w_nbr:
            neighbors[k] = h_nbr
            afs_to_del = {}
            h_addrs = h_nbr.get('address_family', {})
            w_addrs = w_nbr.get('address_family', {})
            for af, h_addr in iteritems(h_addrs):
                if af in w_addrs:
                    afs_to_del[af] = h_addr
            neighbors[k]['address_family'] = afs_to_del
    return neighbors