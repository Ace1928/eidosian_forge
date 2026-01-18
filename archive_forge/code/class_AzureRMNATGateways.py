from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
class AzureRMNATGateways(AzureRMModuleBase):
    """Configuration class for an Azure RM NAT Gateway resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), sku=dict(type='dict', options=sku_spec), idle_timeout_in_minutes=dict(type='int', default=4), zones=dict(type='list', elements='int', choices=[1, 2, 3]), public_ip_addresses=dict(type='list', elements='str'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.parameters = dict()
        self.results = dict(changed=False)
        self.state = None
        self.to_do = Actions.NoAction
        super(AzureRMNATGateways, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                if key == 'id':
                    self.parameters['id'] = kwargs[key]
                elif key == 'location':
                    self.parameters['location'] = kwargs[key]
                elif key == 'idle_timeout_in_minutes':
                    self.parameters['idle_timeout_in_minutes'] = kwargs[key]
                elif key == 'zones':
                    self.parameters['zones'] = kwargs[key]
                elif key == 'public_ip_addresses':
                    if 'public_ip_addresses' not in self.parameters:
                        self.parameters['public_ip_addresses'] = []
                    for resource in kwargs[key]:
                        self.parameters['public_ip_addresses'].append({'id': self.return_resource_id(resource)})
                elif key == 'sku':
                    ev = kwargs[key]
                    if 'name' in ev:
                        if ev['name'] == 'standard':
                            ev['name'] = 'Standard'
                    self.parameters['sku'] = ev
        old_response = None
        response = None
        resource_group = self.get_resource_group(self.resource_group)
        if 'location' not in self.parameters:
            self.parameters['location'] = resource_group.location
        if 'sku' not in self.parameters:
            self.parameters['sku'] = {'name': 'Standard'}
        old_response = self.get_natgateway()
        if not old_response:
            self.log("NAT Gateway instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('NAT Gateway instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            elif self.state == 'present':
                self.log('Need to check if NAT Gateway instance has to be deleted or may be updated')
                self.to_do = Actions.Update
        if self.to_do == Actions.Update:
            if self.parameters['location'] != old_response['location'] or self.check_if_changed('zones', old_response) or self.check_if_changed('idle_timeout_in_minutes', old_response) or self.check_if_changed('public_ip_addresses', old_response) or (self.parameters['sku']['name'] != old_response['sku']['name']):
                self.to_do = Actions.Update
            else:
                self.to_do = Actions.NoAction
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the NAT Gateway instance')
            if self.check_mode:
                self.results['changed'] = True
                self.results['parameters'] = self.parameters
                return self.results
            response = self.create_update_natgateway()
            if not old_response:
                self.results['changed'] = True
            else:
                self.results['changed'] = old_response != response
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('NAT Gateway instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_natgateway()
            while self.get_natgateway():
                time.sleep(20)
        else:
            self.log('NAT Gateway instance unchanged')
            self.results['changed'] = False
            response = old_response
        if response:
            self.results.update(self.format_response(response))
        return self.results

    def create_update_natgateway(self):
        """
        Creates or updates NAT Gateway with the specified configuration.

        :return: deserialized NAT Gateway instance state dictionary
        """
        self.log('Creating / Updating the NAT Gateway instance {0}'.format(self.name))
        try:
            response = self.network_client.nat_gateways.begin_create_or_update(resource_group_name=self.resource_group, nat_gateway_name=self.name, parameters=self.parameters)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the NAT Gateway instance.')
            self.fail('Error creating the NAT Gateway instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_natgateway(self):
        """
        Deletes specified NAT Gateway instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the NAT Gateway instance {0}'.format(self.name))
        try:
            response = self.network_client.nat_gateways.begin_delete(resource_group_name=self.resource_group, nat_gateway_name=self.name)
        except Exception as e:
            self.log('Error attempting to delete the NAT Gateway instance.')
            self.fail('Error deleting the NAT Gateway instance: {0}'.format(str(e)))
        return True

    def get_natgateway(self):
        """
        Gets the properties of the specified NAT Gateway.

        :return: deserialized NAT Gateway instance state dictionary
        """
        self.log('Checking if the NAT Gateway instance {0} is present'.format(self.name))
        found = False
        try:
            response = self.network_client.nat_gateways.get(resource_group_name=self.resource_group, nat_gateway_name=self.name)
            found = True
            self.log('Response : {0}'.format(response))
            self.log('NAT Gateway instance : {0} found'.format(response.name))
        except ResourceNotFoundError as e:
            self.log('Did not find the NAT Gateway instance.')
        if found is True:
            return response.as_dict()
        return False

    def check_if_changed(self, parameter_name, old_response):
        """"
        Compute if there is an update to the resource or not

        :return: True if resource is changed compared to the current one
        """
        if parameter_name in self.parameters and (parameter_name not in old_response or self.parameters[parameter_name] != old_response[parameter_name]):
            return True
        elif parameter_name not in self.parameters and parameter_name in old_response:
            return True
        else:
            return False

    def format_response(self, natgw_dict):
        """
        Build format of the response

        :return dictionary filled with resource data
        """
        id = natgw_dict.get('id')
        id_dict = parse_resource_id(id)
        d = {'id': id, 'name': natgw_dict.get('name'), 'resource_group': id_dict.get('resource_group', self.resource_group), 'location': natgw_dict.get('location')}
        return d

    def return_resource_id(self, resource):
        """
        Build an IP Address resource id from different inputs

        :return string containing the Azure id of the resource
        """
        if is_valid_resource_id(resource):
            return resource
        resource_dict = self.parse_resource_to_dict(resource)
        return format_resource_id(val=resource_dict['name'], subscription_id=resource_dict.get('subscription_id'), namespace='Microsoft.Network', types='publicIPAddresses', resource_group=resource_dict.get('resource_group'))