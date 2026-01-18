from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.bgp_address_family.bgp_address_family import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.bgp_address_family import (
def _process_facts(self, objs):
    """makes data as per the facts after data obtained from parsers"""
    addr_fam_facts = {}
    temp_af = []
    addr_fam_facts['as_number'] = objs['as_number']
    if objs.get('address_family'):
        for kaf, afs in objs['address_family'].items():
            af = {}
            for af_key, afs_val in afs.items():
                if af_key == 'neighbors':
                    temp_neighbor = []
                    for tag, neighbor in afs_val.items():
                        if not neighbor.get('neighbor_address'):
                            neighbor['neighbor_address'] = tag
                        temp_neighbor.append(neighbor)
                    af[af_key] = temp_neighbor
                else:
                    af[af_key] = afs_val
            temp_af.append(af)
    if temp_af:
        addr_fam_facts['address_family'] = temp_af
    return addr_fam_facts