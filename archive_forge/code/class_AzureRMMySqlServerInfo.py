from __future__ import absolute_import, division, print_function
class AzureRMMySqlServerInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.name = None
        self.tags = None
        super(AzureRMMySqlServerInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_mysqlserver_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_mysqlserver_facts' module has been renamed to 'azure_rm_mysqlserver_info'", version=(2.9,))
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
            response = self.mysql_client.servers.get(resource_group_name=self.resource_group, server_name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for MySQL Server.')
        if response and self.has_tags(response.tags, self.tags):
            results.append(self.format_item(response))
        return results

    def list_by_resource_group(self):
        response = None
        results = []
        try:
            response = self.mysql_client.servers.list_by_resource_group(resource_group_name=self.resource_group)
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Could not get facts for MySQL Servers.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_item(item))
        return results

    def format_item(self, item):
        d = item.as_dict()
        d = {'id': d['id'], 'resource_group': self.resource_group, 'name': d['name'], 'sku': d['sku'], 'location': d['location'], 'storage_profile': d['storage_profile'], 'version': d['version'], 'enforce_ssl': d['ssl_enforcement'] == 'Enabled', 'admin_username': d['administrator_login'], 'user_visible_state': d['user_visible_state'], 'fully_qualified_domain_name': d['fully_qualified_domain_name'], 'tags': d.get('tags')}
        return d