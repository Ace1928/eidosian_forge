from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.bgp_address_family import (
def _compare_redist_ospf(self, _parser, w_attr, h_attr):
    """
        Adds and/or removes commands related to
        ospf and ospv3 redistribution
        :param _parser: ospf or ospfv3
        :param w_attr: content of want['redistribute']['ospf']
        :param h_attr:content of have['redistribute']['ospf']
        :return: None
        """
    for wkey, wentry in w_attr.items():
        if wentry != h_attr.pop(wkey, {}):
            if self.state in ['overridden', 'replaced']:
                self.addcmd(wentry, f'redistribute.{_parser}', True)
            self.addcmd(wentry, f'redistribute.{_parser}', False)
    for hkey, hentry in h_attr.items():
        self.addcmd(hentry, 'redistribute.ospf', True)