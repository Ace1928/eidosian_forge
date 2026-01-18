from __future__ import absolute_import, division, print_function
class AzureRMPostgreSQLConfigurationInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), server_name=dict(type='str', required=True), name=dict(type='str'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.server_name = None
        self.name = None
        super(AzureRMPostgreSQLConfigurationInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_postgresqlconfiguration_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_postgresqlconfiguration_facts' module has been renamed to 'azure_rm_postgresqlconfiguration_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name is not None:
            self.results['settings'] = self.get()
        else:
            self.results['settings'] = self.list_by_server()
        return self.results

    def get(self):
        """
        Gets facts of the specified PostgreSQL Configuration.

        :return: deserialized PostgreSQL Configurationinstance state dictionary
        """
        response = None
        results = []
        try:
            response = self.postgresql_client.configurations.get(resource_group_name=self.resource_group, server_name=self.server_name, configuration_name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.fail('Could not get requested setting.')
        if response is not None:
            results.append(self.format_item(response))
        return results

    def list_by_server(self):
        """
        Gets facts of the specified PostgreSQL Configuration.

        :return: deserialized PostgreSQL Configurationinstance state dictionary
        """
        response = None
        results = []
        try:
            response = self.postgresql_client.configurations.list_by_server(resource_group_name=self.resource_group, server_name=self.server_name)
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.fail('Could not get settings for server.')
        if response is not None:
            for item in response:
                results.append(self.format_item(item))
        return results

    def format_item(self, item):
        d = item.as_dict()
        d = {'resource_group': self.resource_group, 'server_name': self.server_name, 'id': d['id'], 'name': d['name'], 'value': d['value'], 'description': d['description'], 'source': d['source']}
        return d