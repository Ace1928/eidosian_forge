from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
class AzureRMVirtualHub(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), location=dict(type='str'), name=dict(type='str', required=True), virtual_wan=dict(type='dict', options=dict(id=dict(type='str'))), vpn_gateway=dict(type='dict', options=dict(id=dict(type='str'))), p2_s_vpn_gateway=dict(type='dict', options=dict(id=dict(type='str'))), express_route_gateway=dict(type='dict', options=dict(id=dict(type='str'))), azure_firewall=dict(type='dict', options=dict(id=dict(type='str'))), security_partner_provider=dict(type='dict', options=dict(id=dict(type='str'))), address_prefix=dict(type='str'), route_table=dict(type='dict', options=dict(routes=dict(type='list', elements='dict', options=dict(address_prefixes=dict(type='list', elements='str'), next_hop_ip_address=dict(type='str'))))), security_provider_name=dict(type='str'), virtual_hub_route_table_v2_s=dict(type='list', elements='dict', options=dict(name=dict(type='str'), routes=dict(type='list', elements='dict', options=dict(destination_type=dict(type='str'), destinations=dict(type='list', elements='str'), next_hop_type=dict(type='str'), next_hops=dict(type='list', elements='str'))), attached_connections=dict(type='list', elements='str'))), sku=dict(type='str'), bgp_connections=dict(type='list', elements='dict', options=dict(id=dict(type='str'))), ip_configurations=dict(type='list', elements='dict', options=dict(id=dict(type='str'))), virtual_router_asn=dict(type='int'), virtual_router_ips=dict(type='list', elements='str'), enable_virtual_router_route_propogation=dict(type='bool'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.location = None
        self.body = {}
        self.results = dict(changed=False)
        self.state = None
        self.to_do = Actions.NoAction
        super(AzureRMVirtualHub, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.body[key] = kwargs[key]
        self.inflate_parameters(self.module_arg_spec, self.body, 0)
        resource_group = self.get_resource_group(self.resource_group)
        if self.location is None:
            self.location = resource_group.location
        self.body['location'] = self.location
        old_response = None
        response = None
        old_response = self.get_resource()
        if not old_response:
            if self.state == 'present':
                self.to_do = Actions.Create
        elif self.state == 'absent':
            self.to_do = Actions.Delete
        else:
            modifiers = {}
            self.create_compare_modifiers(self.module_arg_spec, '', modifiers)
            self.results['modifiers'] = modifiers
            self.results['compare'] = []
            if not self.default_compare(modifiers, self.body, old_response, '', self.results):
                self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.create_update_resource()
        elif self.to_do == Actions.Delete:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_resource()
        else:
            self.results['changed'] = False
            response = old_response
        if response is not None:
            self.results['state'] = response
        return self.results

    def create_update_resource(self):
        try:
            response = self.network_client.virtual_hubs.begin_create_or_update(resource_group_name=self.resource_group, virtual_hub_name=self.name, virtual_hub_parameters=self.body)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the VirtualHub instance.')
            self.fail('Error creating the VirtualHub instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_resource(self):
        try:
            response = self.network_client.virtual_hubs.begin_delete(resource_group_name=self.resource_group, virtual_hub_name=self.name)
        except Exception as e:
            self.log('Error attempting to delete the VirtualHub instance.')
            self.fail('Error deleting the VirtualHub instance: {0}'.format(str(e)))
        return True

    def get_resource(self):
        try:
            response = self.network_client.virtual_hubs.get(resource_group_name=self.resource_group, virtual_hub_name=self.name)
        except ResourceNotFoundError as e:
            return False
        return response.as_dict()