from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.bgp_global import (
def _compare_generic_lists(self, w_attr, h_attr, parser):
    """Handling of gereric list options."""
    for wkey, wentry in iteritems(w_attr):
        if wentry != h_attr.pop(wkey, {}):
            self.addcmd(wentry, parser, False)
    for hkey, hentry in iteritems(h_attr):
        self.addcmd(hentry, parser, True)