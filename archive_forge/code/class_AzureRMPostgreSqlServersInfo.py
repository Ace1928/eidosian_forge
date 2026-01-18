from __future__ import absolute_import, division, print_function
class AzureRMPostgreSqlServersInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.name = None
        self.tags = None
        super(AzureRMPostgreSqlServersInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_postgresqlserver_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_postgresqlserver_facts' module has been renamed to 'azure_rm_postgresqlserver_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.resource_group is not None and self.name is not None:
            self.results['servers'] = self.get()
        elif self.resource_group is not None:
            self.results['servers'] = self.list_by_resource_group()
        return self.results

    def get(self):
        response = None
        results = []
        try:
            response = self.postgresql_client.servers.get(resource_group_name=self.resource_group, server_name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for PostgreSQL Server.')
        if response and self.has_tags(response.tags, self.tags):
            results.append(self.format_item(response))
        return results

    def list_by_resource_group(self):
        response = None
        results = []
        try:
            response = self.postgresql_client.servers.list_by_resource_group(resource_group_name=self.resource_group)
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Could not get facts for PostgreSQL Servers.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_item(item))
        return results

    def format_item(self, item):
        d = item.as_dict()
        d = {'id': d['id'], 'resource_group': self.resource_group, 'name': d['name'], 'sku': d['sku'], 'location': d['location'], 'storage_mb': d['storage_profile']['storage_mb'], 'storage_autogrow': d['storage_profile']['storage_autogrow'] == 'Enabled', 'version': d['version'], 'enforce_ssl': d['ssl_enforcement'] == 'Enabled', 'admin_username': d['administrator_login'], 'user_visible_state': d['user_visible_state'], 'fully_qualified_domain_name': d['fully_qualified_domain_name'], 'geo_redundant_backup': d['storage_profile']['geo_redundant_backup'], 'backup_retention_days': d['storage_profile']['backup_retention_days'], 'tags': d.get('tags')}
        return d