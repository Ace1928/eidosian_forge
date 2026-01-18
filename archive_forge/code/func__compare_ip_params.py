from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
from ansible_collections.arista.eos.plugins.module_utils.network.eos.rm_templates.ospf_interfaces import (
def _compare_ip_params(self, want, have):
    w_params = want.get('ip_params', {})
    h_params = have.get('ip_params', {})
    for afi in ['ipv4', 'ipv6']:
        w_p = w_params.pop(afi, {})
        h_p = h_params.pop(afi, {})
        for k, params in iteritems(w_p):
            if k == 'afi':
                continue
            w = {'afi': afi, k: params}
            h = {'afi': afi, k: h_p.pop(k, None)}
            self.compare(parsers=self.parsers, want={'ip_params': w}, have={'ip_params': h})
        for k, params in iteritems(h_p):
            if k == 'afi':
                continue
            w = {'afi': afi, k: None}
            h = {'afi': afi, k: params}
            self.compare(parsers=self.parsers, want={'ip_params': w}, have={'ip_params': h})