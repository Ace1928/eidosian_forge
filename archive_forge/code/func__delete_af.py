from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.bgp_address_family import (
def _delete_af(self, want, have):
    for as_num, entry in iteritems(want):
        for afi, af_entry in iteritems(entry.get('address_family', {})):
            if have.get('address_family'):
                for hafi, hentry in iteritems(have['address_family']):
                    if hafi == afi:
                        self.commands.append(self._tmplt.render({'as_number': as_num, 'address_family': {'afi': afi}}, 'address_family', True))
        for neigh, neigh_entry in iteritems(entry.get('neighbors', {})):
            if have.get('neighbors'):
                for hneigh, hnentry in iteritems(have['neighbors']):
                    if hneigh == neigh:
                        if not neigh_entry.get('address_family'):
                            self.commands.append(self._tmplt.render({'as_number': as_num, 'neighbors': {'neighbor_address': neigh}}, 'neighbors', True))
                        else:
                            for k in neigh_entry['address_family'].keys():
                                if hnentry.get('address_family') and k in hnentry['address_family'].keys():
                                    self.commands.append(self._tmplt.render({'as_number': as_num, 'neighbors': {'neighbor_address': neigh, 'address_family': {'afi': k}}}, 'neighbors.address_family', True))