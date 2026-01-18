from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.bgp_templates import (
def _compare_path_attribute(self, want, have):
    """Custom handling of neighbor path_attribute
           option.

        :params want: the want neighbor dictionary
        :params have: the have neighbor dictionary
        """
    w_p_attr = want.get('path_attribute', {})
    h_p_attr = have.get('path_attribute', {})
    for wkey, wentry in iteritems(w_p_attr):
        if wentry != h_p_attr.pop(wkey, {}):
            self.addcmd(wentry, 'path_attribute', False)
    for hkey, hentry in iteritems(h_p_attr):
        self.addcmd(hentry, 'path_attribute', True)