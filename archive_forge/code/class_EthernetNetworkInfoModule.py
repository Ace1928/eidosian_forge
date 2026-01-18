from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase
class EthernetNetworkInfoModule(OneViewModuleBase):
    argument_spec = dict(name=dict(type='str'), options=dict(type='list', elements='str'), params=dict(type='dict'))

    def __init__(self):
        super(EthernetNetworkInfoModule, self).__init__(additional_arg_spec=self.argument_spec, supports_check_mode=True)
        self.resource_client = self.oneview_client.ethernet_networks

    def execute_module(self):
        info = {}
        if self.module.params['name']:
            ethernet_networks = self.resource_client.get_by('name', self.module.params['name'])
            if self.module.params.get('options') and ethernet_networks:
                info = self.__gather_optional_info(ethernet_networks[0])
        else:
            ethernet_networks = self.resource_client.get_all(**self.facts_params)
        info['ethernet_networks'] = ethernet_networks
        return dict(changed=False, **info)

    def __gather_optional_info(self, ethernet_network):
        info = {}
        if self.options.get('associatedProfiles'):
            info['enet_associated_profiles'] = self.__get_associated_profiles(ethernet_network)
        if self.options.get('associatedUplinkGroups'):
            info['enet_associated_uplink_groups'] = self.__get_associated_uplink_groups(ethernet_network)
        return info

    def __get_associated_profiles(self, ethernet_network):
        associated_profiles = self.resource_client.get_associated_profiles(ethernet_network['uri'])
        return [self.oneview_client.server_profiles.get(x) for x in associated_profiles]

    def __get_associated_uplink_groups(self, ethernet_network):
        uplink_groups = self.resource_client.get_associated_uplink_groups(ethernet_network['uri'])
        return [self.oneview_client.uplink_sets.get(x) for x in uplink_groups]