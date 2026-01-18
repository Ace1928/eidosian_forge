from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMSqlElasticPoolInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), server_name=dict(type='str', required=True), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.server_name = None
        self.name = None
        self.tags = None
        super(AzureRMSqlElasticPoolInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name is not None:
            self.results['elastic_pool'] = self.get()
        else:
            self.results['elastic_pool'] = self.list_by_server()
        return self.results

    def get(self):
        response = None
        results = []
        try:
            response = self.sql_client.elastic_pools.get(resource_group_name=self.resource_group, server_name=self.server_name, elastic_pool_name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError:
            self.log('Could not get facts for Elastic Pool.')
        if response and self.has_tags(response.tags, self.tags):
            results.append(self.format_item(response))
        return results

    def list_by_server(self):
        response = None
        results = []
        try:
            response = self.sql_client.elastic_pools.list_by_server(resource_group_name=self.resource_group, server_name=self.server_name)
            self.log('Response : {0}'.format(response))
        except Exception:
            self.fail('Could not get facts for elastic pool.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_item(item))
        return results

    def format_item(self, item):
        if not item:
            return None
        d = dict(resource_group=self.resource_group, id=item.id, name=item.name, location=item.location, tags=item.tags, max_size_bytes=item.max_size_bytes, zone_redundant=item.zone_redundant, license_type=item.license_type, maintenance_configuration_id=item.maintenance_configuration_id, per_database_settings=dict(), sku=dict())
        if item.sku is not None:
            d['sku']['name'] = item.sku.name
            d['sku']['tier'] = item.sku.tier
            d['sku']['size'] = item.sku.size
            d['sku']['family'] = item.sku.family
            d['sku']['capacity'] = item.sku.capacity
        if item.per_database_settings is not None:
            d['per_database_settings']['min_capacity'] = item.per_database_settings.min_capacity
            d['per_database_settings']['max_capacity'] = item.per_database_settings.max_capacity
        return d