from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.bgp_global import (
def _compare_neigh_lists(self, want, have):
    for attrib in ['distribute_list', 'filter_list', 'prefix_list', 'route_map']:
        wdict = want.pop(attrib, {})
        hdict = have.pop(attrib, {})
        for key, entry in iteritems(wdict):
            if entry != hdict.pop(key, {}):
                self.addcmd(entry, 'neighbor.{0}'.format(attrib), False)
        for entry in hdict.values():
            self.addcmd(entry, 'neighbor.{0}'.format(attrib), True)