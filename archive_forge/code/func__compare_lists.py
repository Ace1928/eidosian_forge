from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.bgp_address_family import (
def _compare_lists(self, want, have, as_number, afi):
    parsers = ['aggregate_address', 'network.backdoor', 'network.path_limit', 'network.route_map', 'redistribute.metric', 'redistribute.route_map', 'redistribute.table']
    for attrib in ['redistribute', 'networks', 'aggregate_address']:
        wdict = want.pop(attrib, {})
        hdict = have.pop(attrib, {})
        for key, entry in iteritems(wdict):
            if entry != hdict.get(key, {}):
                self.compare(parsers=parsers, want={'as_number': as_number, 'address_family': {'afi': afi, attrib: entry}}, have={'as_number': as_number, 'address_family': {'afi': afi, attrib: hdict.pop(key, {})}})
            hdict.pop(key, {})
        if not wdict and hdict:
            attrib = re.sub('_', '-', attrib)
            attrib = re.sub('networks', 'network', attrib)
            self.commands.append('delete protocols bgp ' + str(as_number) + ' ' + 'address-family ' + afi + ' ' + attrib)
            hdict = {}
        for key, entry in iteritems(hdict):
            self.compare(parsers=parsers, want={}, have={'as_number': as_number, 'address_family': {'afi': afi, attrib: entry}})