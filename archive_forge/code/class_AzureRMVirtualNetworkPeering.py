from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
class AzureRMVirtualNetworkPeering(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), virtual_network=dict(type='raw', required=True), remote_virtual_network=dict(type='raw'), allow_virtual_network_access=dict(type='bool', default=False), allow_forwarded_traffic=dict(type='bool', default=False), allow_gateway_transit=dict(type='bool', default=False), use_remote_gateways=dict(type='bool', default=False), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.virtual_network = None
        self.remote_virtual_network = None
        self.allow_virtual_network_access = None
        self.allow_forwarded_traffic = None
        self.allow_gateway_transit = None
        self.use_remote_gateways = None
        self.results = dict(changed=False)
        super(AzureRMVirtualNetworkPeering, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        to_be_updated = False
        to_be_synced = False
        resource_group = self.get_resource_group(self.resource_group)
        self.virtual_network = self.parse_resource_to_dict(self.virtual_network)
        if self.virtual_network['resource_group'] != self.resource_group:
            self.fail('Resource group of virtual_network is not same as param resource_group')
        self.remote_virtual_network = self.format_vnet_id(self.remote_virtual_network)
        response = self.get_vnet_peering()
        if self.state == 'present':
            if response:
                existing_vnet = self.parse_resource_to_dict(response['id'])
                if existing_vnet['resource_group'] != self.virtual_network['resource_group'] or existing_vnet['name'] != self.virtual_network['name']:
                    self.fail('Cannot update virtual_network of Virtual Network Peering!')
                if response['remote_virtual_network'].lower() != self.remote_virtual_network.lower():
                    self.fail('Cannot update remote_virtual_network of Virtual Network Peering!')
                to_be_updated = self.check_update(response)
                to_be_synced = self.check_sync(response)
            else:
                to_be_updated = True
                virtual_network = self.get_vnet(self.virtual_network['resource_group'], self.virtual_network['name'])
                if not virtual_network:
                    self.fail('Virtual network {0} in resource group {1} does not exist!'.format(self.virtual_network['name'], self.virtual_network['resource_group']))
        elif self.state == 'absent':
            if response:
                self.log('Delete Azure Virtual Network Peering')
                self.results['changed'] = True
                self.results['id'] = response['id']
                if self.check_mode:
                    return self.results
                response = self.delete_vnet_peering()
            else:
                self.log('Azure Virtual Network Peering {0} does not exist in resource group {1}'.format(self.name, self.resource_group))
        if to_be_updated:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.create_or_update_vnet_peering()
            self.results['id'] = response['id']
            to_be_synced = self.check_sync(response)
        if to_be_synced:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            sync_response = self.sync_vnet_peering()
            self.results['peering_sync_level'] = sync_response['peering_sync_level']
        return self.results

    def format_vnet_id(self, vnet):
        if not vnet:
            return vnet
        if isinstance(vnet, dict) and vnet.get('name') and vnet.get('resource_group'):
            remote_vnet_id = format_resource_id(vnet['name'], self.subscription_id, 'Microsoft.Network', 'virtualNetworks', vnet['resource_group'])
        elif isinstance(vnet, str):
            if is_valid_resource_id(vnet):
                remote_vnet_id = vnet
            else:
                remote_vnet_id = format_resource_id(vnet, self.subscription_id, 'Microsoft.Network', 'virtualNetworks', self.resource_group)
        else:
            self.fail('remote_virtual_network could be a valid resource id, dict of name and resource_group, name of virtual network in same resource group.')
        return remote_vnet_id

    def check_sync(self, exisiting_vnet_peering):
        if exisiting_vnet_peering['peering_sync_level'] == 'LocalNotInSync':
            return True
        return False

    def check_update(self, exisiting_vnet_peering):
        if self.allow_forwarded_traffic != exisiting_vnet_peering['allow_forwarded_traffic']:
            return True
        if self.allow_gateway_transit != exisiting_vnet_peering['allow_gateway_transit']:
            return True
        if self.allow_virtual_network_access != exisiting_vnet_peering['allow_virtual_network_access']:
            return True
        if self.use_remote_gateways != exisiting_vnet_peering['use_remote_gateways']:
            return True
        return False

    def get_vnet(self, resource_group, vnet_name):
        """
        Get Azure Virtual Network
        :return: deserialized Azure Virtual Network
        """
        self.log('Get the Azure Virtual Network {0}'.format(vnet_name))
        vnet = self.network_client.virtual_networks.get(resource_group, vnet_name)
        if vnet:
            results = virtual_network_to_dict(vnet)
            return results
        return False

    def sync_vnet_peering(self):
        """
        Creates or Update Azure Virtual Network Peering.

        :return: deserialized Azure Virtual Network Peering instance state dictionary
        """
        self.log('Creating or Updating the Azure Virtual Network Peering {0}'.format(self.name))
        vnet_id = format_resource_id(self.virtual_network['name'], self.subscription_id, 'Microsoft.Network', 'virtualNetworks', self.virtual_network['resource_group'])
        peering = self.network_models.VirtualNetworkPeering(id=vnet_id, name=self.name, remote_virtual_network=self.network_models.SubResource(id=self.remote_virtual_network), allow_virtual_network_access=self.allow_virtual_network_access, allow_gateway_transit=self.allow_gateway_transit, allow_forwarded_traffic=self.allow_forwarded_traffic, use_remote_gateways=self.use_remote_gateways)
        try:
            response = self.network_client.virtual_network_peerings.begin_create_or_update(self.resource_group, self.virtual_network['name'], self.name, peering, sync_remote_address_space=True)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
            return vnetpeering_to_dict(response)
        except Exception as exc:
            self.fail('Error creating Azure Virtual Network Peering: {0}.'.format(exc.message))

    def create_or_update_vnet_peering(self):
        """
        Creates or Update Azure Virtual Network Peering.

        :return: deserialized Azure Virtual Network Peering instance state dictionary
        """
        self.log('Creating or Updating the Azure Virtual Network Peering {0}'.format(self.name))
        vnet_id = format_resource_id(self.virtual_network['name'], self.subscription_id, 'Microsoft.Network', 'virtualNetworks', self.virtual_network['resource_group'])
        peering = self.network_models.VirtualNetworkPeering(id=vnet_id, name=self.name, remote_virtual_network=self.network_models.SubResource(id=self.remote_virtual_network), allow_virtual_network_access=self.allow_virtual_network_access, allow_gateway_transit=self.allow_gateway_transit, allow_forwarded_traffic=self.allow_forwarded_traffic, use_remote_gateways=self.use_remote_gateways)
        try:
            response = self.network_client.virtual_network_peerings.begin_create_or_update(self.resource_group, self.virtual_network['name'], self.name, peering)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
            return vnetpeering_to_dict(response)
        except Exception as exc:
            self.fail('Error creating Azure Virtual Network Peering: {0}.'.format(exc.message))

    def delete_vnet_peering(self):
        """
        Deletes the specified Azure Virtual Network Peering

        :return: True
        """
        self.log('Deleting Azure Virtual Network Peering {0}'.format(self.name))
        try:
            poller = self.network_client.virtual_network_peerings.begin_delete(self.resource_group, self.virtual_network['name'], self.name)
            self.get_poller_result(poller)
            return True
        except Exception as e:
            self.fail('Error deleting the Azure Virtual Network Peering: {0}'.format(e.message))
            return False

    def get_vnet_peering(self):
        """
        Gets the Virtual Network Peering.

        :return: deserialized Virtual Network Peering
        """
        self.log('Checking if Virtual Network Peering {0} is present'.format(self.name))
        try:
            response = self.network_client.virtual_network_peerings.get(self.resource_group, self.virtual_network['name'], self.name)
            self.log('Response : {0}'.format(response))
            return vnetpeering_to_dict(response)
        except ResourceNotFoundError:
            self.log('Did not find the Virtual Network Peering.')
            return False