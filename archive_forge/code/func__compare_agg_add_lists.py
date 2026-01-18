from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.bgp_address_family import (
def _compare_agg_add_lists(self, w_attr, h_attr):
    """Handling of agg_add list options."""
    for wkey, wentry in w_attr.items():
        if wentry != h_attr.pop(wkey, {}):
            self.addcmd(wentry, 'aggregate_addresses', False)
    for hkey, hentry in h_attr.items():
        self.addcmd(hentry, 'aggregate_addresses', True)