from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
from ansible_collections.arista.eos.plugins.module_utils.network.eos.rm_templates.logging_global import (
def _hosts_compare(self, want, have):
    host_want = want.pop('hosts', {})
    host_have = have.pop('hosts', {})
    for name, entry in iteritems(host_want):
        h = {}
        if host_have:
            h = {'hosts': host_have.pop(name, {})}
        self.compare(parsers='host', want={'hosts': entry}, have=h)
    for name, entry in iteritems(host_have):
        self.compare(parsers='host', want={}, have={'hosts': entry})