from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.route_maps import (
def _compare_entries(self, want, have):
    for wk, wentry in iteritems(want):
        hentry = have.pop(wk, {})
        self.compare(parsers=self.parsers, want=wentry, have=hentry)