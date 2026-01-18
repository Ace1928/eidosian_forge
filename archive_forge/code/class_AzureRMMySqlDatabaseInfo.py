from __future__ import absolute_import, division, print_function
class AzureRMMySqlDatabaseInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), server_name=dict(type='str', required=True), name=dict(type='str'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.server_name = None
        self.name = None
        super(AzureRMMySqlDatabaseInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_mysqldatabase_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_mysqldatabase_facts' module has been renamed to 'azure_rm_mysqldatabase_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.resource_group is not None and self.server_name is not None and (self.name is not None):
            self.results['databases'] = self.get()
        elif self.resource_group is not None and self.server_name is not None:
            self.results['databases'] = self.list_by_server()
        return self.results

    def get(self):
        response = None
        results = []
        try:
            response = self.mysql_client.databases.get(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for Databases.')
        if response is not None:
            results.append(self.format_item(response))
        return results

    def list_by_server(self):
        response = None
        results = []
        try:
            response = self.mysql_client.databases.list_by_server(resource_group_name=self.resource_group, server_name=self.server_name)
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.fail('Error listing for server {0} - {1}'.format(self.server_name, str(e)))
        if response is not None:
            for item in response:
                results.append(self.format_item(item))
        return results

    def format_item(self, item):
        d = item.as_dict()
        d = {'resource_group': self.resource_group, 'server_name': self.server_name, 'name': d['name'], 'charset': d['charset'], 'collation': d['collation']}
        return d