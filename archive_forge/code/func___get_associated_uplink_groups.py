from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase
def __get_associated_uplink_groups(self, ethernet_network):
    uplink_groups = self.resource_client.get_associated_uplink_groups(ethernet_network['uri'])
    return [self.oneview_client.uplink_sets.get(x) for x in uplink_groups]