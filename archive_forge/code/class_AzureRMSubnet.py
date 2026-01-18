from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, CIDR_PATTERN, azure_id_to_dict, format_resource_id
class AzureRMSubnet(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), virtual_network_name=dict(type='str', required=True, aliases=['virtual_network']), address_prefix_cidr=dict(type='str', aliases=['address_prefix']), address_prefixes_cidr=dict(type='list', aliases=['address_prefixes'], elements='str'), security_group=dict(type='raw', aliases=['security_group_name']), route_table=dict(type='raw'), service_endpoints=dict(type='list', elements='dict', options=dict(service=dict(type='str', required=True), locations=dict(type='list', elements='str'))), private_endpoint_network_policies=dict(type='str', default='Enabled', choices=['Enabled', 'Disabled']), private_link_service_network_policies=dict(type='str', default='Enabled', choices=['Enabled', 'Disabled']), delegations=dict(type='list', elements='dict', options=delegations_spec), nat_gateway=dict(type='str'))
        mutually_exclusive = [['address_prefix_cidr', 'address_prefixes_cidr']]
        self.results = dict(changed=False, state=dict())
        self.resource_group = None
        self.name = None
        self.state = None
        self.virtual_network_name = None
        self.address_prefix_cidr = None
        self.address_prefixes_cidr = None
        self.security_group = None
        self.route_table = None
        self.service_endpoints = None
        self.private_link_service_network_policies = None
        self.private_endpoint_network_policies = None
        self.delegations = None
        self.nat_gateway = None
        super(AzureRMSubnet, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        nsg = None
        subnet = None
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.delegations and len(self.delegations) > 1:
            self.fail('Only one delegation is supported for a subnet')
        if self.address_prefix_cidr and (not CIDR_PATTERN.match(self.address_prefix_cidr)):
            self.fail('Invalid address_prefix_cidr value {0}'.format(self.address_prefix_cidr))
        nsg = dict()
        if self.security_group:
            nsg = self.parse_nsg()
        nat_gateway = self.build_nat_gateway_id(self.nat_gateway)
        route_table = dict()
        if self.route_table:
            route_table = self.parse_resource_to_dict(self.route_table)
            self.route_table = format_resource_id(val=route_table['name'], subscription_id=route_table['subscription_id'], namespace='Microsoft.Network', types='routeTables', resource_group=route_table['resource_group'])
        results = dict()
        changed = False
        try:
            self.log('Fetching subnet {0}'.format(self.name))
            subnet = self.network_client.subnets.get(self.resource_group, self.virtual_network_name, self.name)
            self.check_provisioning_state(subnet, self.state)
            results = subnet_to_dict(subnet)
            if self.state == 'present':
                if self.private_endpoint_network_policies is not None:
                    if results['private_endpoint_network_policies'] != self.private_endpoint_network_policies:
                        self.log('CHANGED: subnet {0} private_endpoint_network_policies'.format(self.private_endpoint_network_policies))
                        changed = True
                        results['private_endpoint_network_policies'] = self.private_endpoint_network_policies
                else:
                    subnet['private_endpoint_network_policies'] = results['private_endpoint_network_policies']
                if self.private_link_service_network_policies is not None:
                    if results['private_link_service_network_policies'] != self.private_link_service_network_policies:
                        self.log('CHANGED: subnet {0} private_link_service_network_policies'.format(self.private_link_service_network_policies))
                        changed = True
                        results['private_link_service_network_policies'] = self.private_link_service_network_policies
                else:
                    subnet['private_link_service_network_policies'] = results['private_link_service_network_policies']
                if self.address_prefix_cidr and results['address_prefix'] != self.address_prefix_cidr:
                    self.log('CHANGED: subnet {0} address_prefix_cidr'.format(self.name))
                    changed = True
                    results['address_prefix'] = self.address_prefix_cidr
                if self.address_prefixes_cidr and results['address_prefixes'] != self.address_prefixes_cidr:
                    self.log('CHANGED: subnet {0} address_prefixes_cidr'.format(self.name))
                    changed = True
                    results['address_prefixes'] = self.address_prefixes_cidr
                if self.security_group is not None and results['network_security_group'].get('id') != nsg.get('id'):
                    self.log('CHANGED: subnet {0} network security group'.format(self.name))
                    changed = True
                    results['network_security_group']['id'] = nsg.get('id')
                    results['network_security_group']['name'] = nsg.get('name')
                if self.route_table is not None:
                    if self.route_table != results['route_table'].get('id'):
                        changed = True
                        results['route_table']['id'] = self.route_table
                        self.log('CHANGED: subnet {0} route_table to {1}'.format(self.name, route_table.get('name')))
                elif results['route_table'].get('id') is not None:
                    changed = True
                    results['route_table']['id'] = None
                    self.log('CHANGED: subnet {0} will dissociate to route_table {1}'.format(self.name, route_table.get('name')))
                if self.service_endpoints or self.service_endpoints == []:
                    oldd = {}
                    for item in self.service_endpoints:
                        name = item['service']
                        locations = item.get('locations') or []
                        oldd[name] = {'service': name, 'locations': locations.sort()}
                    newd = {}
                    if 'service_endpoints' in results:
                        for item in results['service_endpoints']:
                            name = item['service']
                            locations = item.get('locations') or []
                            newd[name] = {'service': name, 'locations': locations.sort()}
                    if newd != oldd:
                        changed = True
                        results['service_endpoints'] = self.service_endpoints
                if self.delegations:
                    oldde = {}
                    for item in self.delegations:
                        name = item['name']
                        serviceName = item['serviceName']
                        actions = item.get('actions') or []
                        oldde[name] = {'name': name, 'serviceName': serviceName, 'actions': actions.sort()}
                    newde = {}
                    if 'delegations' in results:
                        for item in results['delegations']:
                            name = item['name']
                            serviceName = item['serviceName']
                            actions = item.get('actions') or []
                            newde[name] = {'name': name, 'serviceName': serviceName, 'actions': actions.sort()}
                    if newde != oldde:
                        changed = True
                        results['delegations'] = self.delegations
                if nat_gateway is not None:
                    if nat_gateway != results['nat_gateway']:
                        changed = True
                        results['nat_gateway'] = nat_gateway
                elif results['nat_gateway'] is not None:
                    changed = True
                    results['nat_gateway'] = None
            elif self.state == 'absent':
                changed = True
        except ResourceNotFoundError:
            if self.state == 'present':
                changed = True
        self.results['changed'] = changed
        self.results['state'] = results
        if not self.check_mode:
            if self.state == 'present' and changed:
                if not subnet:
                    if not self.address_prefix_cidr and (not self.address_prefixes_cidr):
                        self.fail('address_prefix_cidr or address_prefixes_cidr is not set')
                    self.log('Creating subnet {0}'.format(self.name))
                    subnet = self.network_models.Subnet(address_prefix=self.address_prefix_cidr, address_prefixes=self.address_prefixes_cidr)
                    if nsg:
                        subnet.network_security_group = self.network_models.NetworkSecurityGroup(id=nsg.get('id'))
                    if self.route_table:
                        subnet.route_table = self.network_models.RouteTable(id=self.route_table)
                    if self.service_endpoints:
                        subnet.service_endpoints = self.service_endpoints
                    if self.private_endpoint_network_policies:
                        subnet.private_endpoint_network_policies = self.private_endpoint_network_policies
                    if self.private_link_service_network_policies:
                        subnet.private_link_service_network_policies = self.private_link_service_network_policies
                    if self.delegations:
                        subnet.delegations = self.delegations
                    if nat_gateway:
                        subnet.nat_gateway = self.network_models.SubResource(id=nat_gateway)
                else:
                    self.log('Updating subnet {0}'.format(self.name))
                    subnet = self.network_models.Subnet(address_prefix=results['address_prefix'], address_prefixes=results['address_prefixes'])
                    if results['network_security_group'].get('id') is not None:
                        subnet.network_security_group = self.network_models.NetworkSecurityGroup(id=results['network_security_group'].get('id'))
                    if results['route_table'].get('id') is not None:
                        subnet.route_table = self.network_models.RouteTable(id=results['route_table'].get('id'))
                    if results.get('service_endpoints') is not None:
                        subnet.service_endpoints = results['service_endpoints']
                    if results.get('private_link_service_network_policies') is not None:
                        subnet.private_link_service_network_policies = results['private_link_service_network_policies']
                    if results.get('private_endpoint_network_policies') is not None:
                        subnet.private_endpoint_network_policies = results['private_endpoint_network_policies']
                    if results.get('delegations') is not None:
                        subnet.delegations = results['delegations']
                    if results.get('nat_gateway') is not None:
                        subnet.nat_gateway = self.network_models.SubResource(id=results['nat_gateway'])
                self.results['state'] = self.create_or_update_subnet(subnet)
            elif self.state == 'absent' and changed:
                self.delete_subnet()
                self.results['state']['status'] = 'Deleted'
        return self.results

    def create_or_update_subnet(self, subnet):
        try:
            poller = self.network_client.subnets.begin_create_or_update(self.resource_group, self.virtual_network_name, self.name, subnet)
            new_subnet = self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error creating or updating subnet {0} - {1}'.format(self.name, str(exc)))
        self.check_provisioning_state(new_subnet)
        return subnet_to_dict(new_subnet)

    def delete_subnet(self):
        self.log('Deleting subnet {0}'.format(self.name))
        try:
            poller = self.network_client.subnets.begin_delete(self.resource_group, self.virtual_network_name, self.name)
            result = self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error deleting subnet {0} - {1}'.format(self.name, str(exc)))
        return result

    def parse_nsg(self):
        nsg = self.security_group
        resource_group = self.resource_group
        if isinstance(self.security_group, dict):
            nsg = self.security_group.get('name')
            resource_group = self.security_group.get('resource_group', self.resource_group)
        id = format_resource_id(val=nsg, subscription_id=self.subscription_id, namespace='Microsoft.Network', types='networkSecurityGroups', resource_group=resource_group)
        name = azure_id_to_dict(id).get('name')
        return dict(id=id, name=name)

    def build_nat_gateway_id(self, resource):
        """
        Common method to build a resource id from different inputs
        """
        if resource is None:
            return None
        if is_valid_resource_id(resource):
            return resource
        resource_dict = self.parse_resource_to_dict(resource)
        return format_resource_id(val=resource_dict['name'], subscription_id=resource_dict.get('subscription_id'), namespace='Microsoft.Network', types='natGateways', resource_group=resource_dict.get('resource_group'))