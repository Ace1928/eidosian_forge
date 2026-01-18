from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMSubnetInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), virtual_network_name=dict(type='str', required=True), name=dict(type='str'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.virtual_network_name = None
        self.name = None
        super(AzureRMSubnetInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_subnet_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_subnet_facts' module has been renamed to 'azure_rm_subnet_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name is not None:
            self.results['subnets'] = self.get()
        else:
            self.results['subnets'] = self.list()
        return self.results

    def get(self):
        response = None
        results = []
        try:
            response = self.network_client.subnets.get(resource_group_name=self.resource_group, virtual_network_name=self.virtual_network_name, subnet_name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.fail('Could not get facts for Subnet.')
        if response is not None:
            results.append(self.format_response(response))
        return results

    def list(self):
        response = None
        results = []
        try:
            response = self.network_client.subnets.get(resource_group_name=self.resource_group, virtual_network_name=self.virtual_network_name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.fail('Could not get facts for Subnet.')
        if response is not None:
            for item in response:
                results.append(self.format_item(item))
        return results

    def format_response(self, item):
        d = item.as_dict()
        d = {'resource_group': self.resource_group, 'virtual_network_name': self.parse_resource_to_dict(d.get('id')).get('name'), 'name': d.get('name'), 'id': d.get('id'), 'address_prefix_cidr': d.get('address_prefix'), 'address_prefixes_cidr': d.get('address_prefixes'), 'route_table': d.get('route_table', {}).get('id'), 'security_group': d.get('network_security_group', {}).get('id'), 'provisioning_state': d.get('provisioning_state'), 'service_endpoints': d.get('service_endpoints'), 'private_endpoint_network_policies': d.get('private_endpoint_network_policies'), 'private_link_service_network_policies': d.get('private_link_service_network_policies'), 'delegations': d.get('delegations'), 'nat_gateway': d.get('nat_gateway', {}).get('id')}
        return d