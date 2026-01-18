from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, \
class AzureRMVirtualNetworkLink(AzureRMModuleBase):

    def __init__(self):
        _load_params()
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), zone_name=dict(type='str', required=True), virtual_network=dict(type='str'), state=dict(choices=['present', 'absent'], default='present', type='str'), registration_enabled=dict(type='bool', default=False))
        required_if = [('state', 'present', ['virtual_network'])]
        self.results = dict(changed=False, state=dict())
        self.resource_group = None
        self.name = None
        self.zone_name = None
        self.virtual_network = None
        self.registration_enabled = None
        self.state = None
        self.tags = None
        self.log_path = None
        self.log_mode = None
        super(AzureRMVirtualNetworkLink, self).__init__(self.module_arg_spec, required_if=required_if, supports_tags=True, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        changed = False
        results = dict()
        zone = None
        virtual_network_link_old = None
        virtual_network_link_new = None
        self.get_resource_group(self.resource_group)
        if self.virtual_network:
            virtual_network = self.parse_resource_to_dict(self.virtual_network)
            self.virtual_network = format_resource_id(val=virtual_network['name'], subscription_id=virtual_network['subscription_id'], namespace='Microsoft.Network', types='virtualNetworks', resource_group=virtual_network['resource_group'])
        self.log('Fetching Private DNS zone {0}'.format(self.zone_name))
        zone = self.private_dns_client.private_zones.get(self.resource_group, self.zone_name)
        if not zone:
            self.fail('The zone {0} does not exist in the resource group {1}'.format(self.zone_name, self.resource_group))
        try:
            self.log('Fetching Virtual network link {0}'.format(self.name))
            virtual_network_link_old = self.private_dns_client.virtual_network_links.get(self.resource_group, self.zone_name, self.name)
            results = self.vnetlink_to_dict(virtual_network_link_old)
            if self.state == 'present':
                changed = False
                update_tags, results['tags'] = self.update_tags(results['tags'])
                if update_tags:
                    changed = True
                self.tags = results['tags']
                if self.registration_enabled != results['registration_enabled']:
                    changed = True
                    results['registration_enabled'] = self.registration_enabled
            elif self.state == 'absent':
                changed = True
        except ResourceNotFoundError:
            if self.state == 'present':
                changed = True
            else:
                changed = False
        self.results['changed'] = changed
        self.results['state'] = results
        if self.check_mode:
            return self.results
        if changed:
            if self.state == 'present':
                virtual_network_link_new = self.private_dns_models.VirtualNetworkLink(location='global', registration_enabled=self.registration_enabled)
                if self.virtual_network:
                    virtual_network_link_new.virtual_network = self.network_models.VirtualNetwork(id=self.virtual_network)
                if self.tags:
                    virtual_network_link_new.tags = self.tags
                self.results['state'] = self.create_or_update_network_link(virtual_network_link_new)
            elif self.state == 'absent':
                self.delete_network_link()
                self.results['state'] = 'Deleted'
        return self.results

    def create_or_update_network_link(self, virtual_network_link):
        try:
            response = self.private_dns_client.virtual_network_links.begin_create_or_update(resource_group_name=self.resource_group, private_zone_name=self.zone_name, virtual_network_link_name=self.name, parameters=virtual_network_link)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.fail('Error creating or updating virtual network link {0} - {1}'.format(self.name, str(exc)))
        return self.vnetlink_to_dict(response)

    def delete_network_link(self):
        try:
            response = self.private_dns_client.virtual_network_links.begin_delete(resource_group_name=self.resource_group, private_zone_name=self.zone_name, virtual_network_link_name=self.name)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.fail('Error deleting virtual network link {0} - {1}'.format(self.name, str(exc)))
        return response

    def vnetlink_to_dict(self, virtualnetworklink):
        result = virtualnetworklink.as_dict()
        result['tags'] = virtualnetworklink.tags
        return result