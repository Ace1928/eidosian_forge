from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase
def __gather_optional_info(self, ethernet_network):
    info = {}
    if self.options.get('associatedProfiles'):
        info['enet_associated_profiles'] = self.__get_associated_profiles(ethernet_network)
    if self.options.get('associatedUplinkGroups'):
        info['enet_associated_uplink_groups'] = self.__get_associated_uplink_groups(ethernet_network)
    return info