from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.bgp_global import (
def _compare_bgp_params(self, want, have):
    parsers = ['bgp_params.always_compare_med', 'bgp_params.bestpath.as_path', 'bgp_params.bestpath.compare_routerid', 'bgp_params.bestpath.med', 'bgp_params.cluster_id', 'bgp_params.confederation', 'bgp_params.dampening_half_life', 'bgp_params.dampening_max_suppress_time', 'bgp_params.dampening_re_use', 'bgp_params.dampening_start_suppress_time', 'bgp_params.default', 'bgp_params.deterministic_med', 'bgp_params.disbale_network_import_check', 'bgp_params.enforce_first_as', 'bgp_params.graceful_restart', 'bgp_params.log_neighbor_changes', 'bgp_params.no_client_to_client_reflection', 'bgp_params.no_fast_external_failover', 'bgp_params.routerid', 'bgp_params.scan_time']
    wbgp = want.pop('bgp_params', {})
    hbgp = have.pop('bgp_params', {})
    for name, entry in iteritems(wbgp):
        if name == 'confederation':
            if entry != hbgp.pop(name, {}):
                self.addcmd({'as_number': want['as_number'], 'bgp_params': {name: entry}}, 'bgp_params.confederation', False)
        elif name == 'distance':
            if entry != hbgp.pop(name, {}):
                distance_parsers = ['bgp_params.distance.global', 'bgp_params.distance.prefix']
                for distance_type in entry:
                    self.compare(parsers=distance_parsers, want={'as_number': want['as_number'], 'bgp_params': {name: distance_type}}, have={'as_number': want['as_number'], 'bgp_params': {name: hbgp.pop(name, {})}})
        else:
            self.compare(parsers=parsers, want={'as_number': want['as_number'], 'bgp_params': {name: entry}}, have={'as_number': want['as_number'], 'bgp_params': {name: hbgp.pop(name, {})}})
    if not wbgp and hbgp:
        self.commands.append('delete protocols bgp ' + str(have['as_number']) + ' parameters')
        hbgp = {}
    for name, entry in iteritems(hbgp):
        if name == 'confederation':
            self.commands.append('delete protocols bgp ' + str(have['as_number']) + ' parameters confederation')
        elif name == 'distance':
            distance_parsers = ['bgp_params.distance.global', 'bgp_params.distance.prefix']
            self.compare(parsers=distance_parsers, want={}, have={'as_number': have['as_number'], 'bgp_params': {name: entry[0]}})
        else:
            self.compare(parsers=parsers, want={}, have={'as_number': have['as_number'], 'bgp_params': {name: entry}})