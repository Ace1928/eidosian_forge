from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
class AzureRMCosmosDBAccount(AzureRMModuleBase):
    """Configuration class for an Azure RM Database Account resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), kind=dict(type='str', choices=['global_document_db', 'mongo_db', 'parse']), consistency_policy=dict(type='dict', options=dict(default_consistency_level=dict(type='str', choices=['eventual', 'session', 'bounded_staleness', 'strong', 'consistent_prefix']), max_staleness_prefix=dict(type='int'), max_interval_in_seconds=dict(type='int'))), geo_rep_locations=dict(type='list', elements='dict', options=dict(name=dict(type='str', required=True), failover_priority=dict(type='int', required=True))), database_account_offer_type=dict(type='str'), enable_free_tier=dict(type='bool', default=False), ip_range_filter=dict(type='str'), ip_rules=dict(type='list', elements='str'), is_virtual_network_filter_enabled=dict(type='bool'), enable_automatic_failover=dict(type='bool'), enable_cassandra=dict(type='bool'), enable_table=dict(type='bool'), enable_gremlin=dict(type='bool'), mongo_version=dict(type='str'), public_network_access=dict(type='str', default='Enabled', choices=['Enabled', 'Disabled']), virtual_network_rules=dict(type='list', elements='dict', options=dict(subnet=dict(type='raw', required=True), ignore_missing_v_net_service_endpoint=dict(type='bool'))), enable_multiple_write_locations=dict(type='bool'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.parameters = dict()
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.to_do = Actions.NoAction
        super(AzureRMCosmosDBAccount, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.parameters[key] = kwargs[key]
        kind = self.parameters.get('kind')
        if kind == 'global_document_db':
            self.parameters['kind'] = 'GlobalDocumentDB'
        elif kind == 'mongo_db':
            self.parameters['kind'] = 'MongoDB'
        elif kind == 'parse':
            self.parameters['kind'] = 'Parse'
        ip_range_filter = self.parameters.pop('ip_range_filter', None)
        ip_rules = self.parameters.pop('ip_rules', [])
        if ip_range_filter:
            self.parameters['ip_rules'] = [{'ip_address_or_range': ip} for ip in ip_range_filter.split(',')]
        if ip_rules:
            self.parameters['ip_rules'] = [{'ip_address_or_range': ip} for ip in ip_rules]
        dict_camelize(self.parameters, ['consistency_policy', 'default_consistency_level'], True)
        dict_rename(self.parameters, ['geo_rep_locations', 'name'], 'location_name')
        dict_rename(self.parameters, ['geo_rep_locations'], 'locations')
        self.parameters['capabilities'] = []
        if self.parameters.pop('enable_cassandra', False):
            self.parameters['capabilities'].append({'name': 'EnableCassandra'})
        if self.parameters.pop('enable_table', False):
            self.parameters['capabilities'].append({'name': 'EnableTable'})
        if self.parameters.pop('enable_gremlin', False):
            self.parameters['capabilities'].append({'name': 'EnableGremlin'})
        mongo_version = self.parameters.pop('mongo_version', None)
        if kind == 'mongo_db' and mongo_version is not None:
            self.parameters['api_properties'] = dict()
            self.parameters['api_properties']['server_version'] = mongo_version
        for rule in self.parameters.get('virtual_network_rules', []):
            subnet = rule.pop('subnet')
            if isinstance(subnet, dict):
                virtual_network_name = subnet.get('virtual_network_name')
                subnet_name = subnet.get('subnet_name')
                resource_group_name = subnet.get('resource_group', self.resource_group)
                template = '/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/virtualNetworks/{2}/subnets/{3}'
                subnet = template.format(self.subscription_id, resource_group_name, virtual_network_name, subnet_name)
            rule['id'] = subnet
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(CosmosDBManagementClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        resource_group = self.get_resource_group(self.resource_group)
        if 'location' not in self.parameters:
            self.parameters['location'] = resource_group.location
        old_response = self.get_databaseaccount()
        if not old_response:
            self.log("Database Account instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('Database Account instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            elif self.state == 'present':
                old_response['locations'] = old_response['failover_policies']
                if not default_compare(self.parameters, old_response, '', self.results):
                    self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the Database Account instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_databaseaccount()
            self.results['changed'] = True
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('Database Account instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_databaseaccount()
        else:
            self.log('Database Account instance unchanged')
            self.results['changed'] = False
            response = old_response
        if self.state == 'present':
            self.results.update({'id': response.get('id', None)})
        return self.results

    def create_update_databaseaccount(self):
        """
        Creates or updates Database Account with the specified configuration.

        :return: deserialized Database Account instance state dictionary
        """
        self.log('Creating / Updating the Database Account instance {0}'.format(self.name))
        try:
            response = self.mgmt_client.database_accounts.begin_create_or_update(resource_group_name=self.resource_group, account_name=self.name, create_update_parameters=self.parameters)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the Database Account instance.')
            self.fail('Error creating the Database Account instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_databaseaccount(self):
        """
        Deletes specified Database Account instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the Database Account instance {0}'.format(self.name))
        try:
            response = self.mgmt_client.database_accounts.begin_delete(resource_group_name=self.resource_group, account_name=self.name)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as e:
            self.log('Error attempting to delete the Database Account instance.')
            self.fail('Error deleting the Database Account instance: {0}'.format(str(e)))
        return True

    def get_databaseaccount(self):
        """
        Gets the properties of the specified Database Account.

        :return: deserialized Database Account instance state dictionary
        """
        self.log('Checking if the Database Account instance {0} is present'.format(self.name))
        found = False
        try:
            response = self.mgmt_client.database_accounts.get(resource_group_name=self.resource_group, account_name=self.name)
            if not response:
                return False
            found = True
            self.log('Response : {0}'.format(response))
            self.log('Database Account instance : {0} found'.format(response.name))
        except ResourceNotFoundError as e:
            self.log('Did not find the Database Account instance.')
        if found is True:
            return response.as_dict()
        return False