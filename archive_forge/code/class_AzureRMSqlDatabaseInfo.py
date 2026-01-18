from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMSqlDatabaseInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), server_name=dict(type='str', required=True), name=dict(type='str'), elastic_pool_name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.server_name = None
        self.name = None
        self.elastic_pool_name = None
        self.tags = None
        super(AzureRMSqlDatabaseInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_sqldatabase_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_sqldatabase_facts' module has been renamed to 'azure_rm_sqldatabase_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name is not None:
            self.results['databases'] = self.get()
        elif self.elastic_pool_name is not None:
            self.results['databases'] = self.list_by_elastic_pool()
        else:
            self.results['databases'] = self.list_by_server()
        return self.results

    def get(self):
        response = None
        results = []
        try:
            response = self.sql_client.databases.get(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError:
            self.log('Could not get facts for Databases.')
        if response and self.has_tags(response.tags, self.tags):
            results.append(self.format_item(response))
        return results

    def list_by_elastic_pool(self):
        response = None
        results = []
        try:
            response = self.sql_client.databases.list_by_elastic_pool(resource_group_name=self.resource_group, server_name=self.server_name, elastic_pool_name=self.elastic_pool_name)
            self.log('Response : {0}'.format(response))
        except Exception:
            self.fail('Could not get facts for Databases.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_item(item))
        return results

    def list_by_server(self):
        response = None
        results = []
        try:
            response = self.sql_client.databases.list_by_server(resource_group_name=self.resource_group, server_name=self.server_name)
            self.log('Response : {0}'.format(response))
        except Exception:
            self.fail('Could not get facts for Databases.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_item(item))
        return results

    def format_item(self, item):
        d = item.as_dict()
        d = {'resource_group': self.resource_group, 'id': d.get('id', None), 'name': d.get('name', None), 'location': d.get('location', None), 'tags': d.get('tags', None), 'sku': {'name': d.get('current_service_objective_name', None), 'tier': d.get('sku', {}).get('tier', None), 'capacity': d.get('sku', {}).get('capacity', None)}, 'kind': d.get('kind', None), 'collation': d.get('collation', None), 'status': d.get('status', None), 'zone_redundant': d.get('zone_redundant', None), 'earliest_restore_date': d.get('earliest_restore_date', None)}
        return d