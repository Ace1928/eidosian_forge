from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase
def __get_associated_profiles(self, ethernet_network):
    associated_profiles = self.resource_client.get_associated_profiles(ethernet_network['uri'])
    return [self.oneview_client.server_profiles.get(x) for x in associated_profiles]