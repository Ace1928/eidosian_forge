from __future__ import absolute_import, division, print_function
import time
class AzureRMMySqlServers(AzureRMModuleBase):
    """Configuration class for an Azure RM MySQL Server resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), sku=dict(type='dict'), location=dict(type='str'), storage_profile=dict(type='dict', options=storage_profile_spec), version=dict(type='str', choices=['5.7', '8.0']), enforce_ssl=dict(type='bool', default=False), create_mode=dict(type='str', default='Default'), admin_username=dict(type='str'), restarted=dict(type='bool', default=False), admin_password=dict(type='str', no_log=True), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.parameters = dict()
        self.tags = None
        self.restarted = False
        self.results = dict(changed=False)
        self.state = None
        self.to_do = Actions.NoAction
        super(AzureRMMySqlServers, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                if key == 'sku':
                    ev = kwargs[key]
                    if 'tier' in ev:
                        if ev['tier'] == 'basic':
                            ev['tier'] = 'Basic'
                        elif ev['tier'] == 'standard':
                            ev['tier'] = 'Standard'
                    self.parameters['sku'] = ev
                elif key == 'location':
                    self.parameters['location'] = kwargs[key]
                elif key == 'storage_profile':
                    self.parameters.setdefault('properties', {})['storage_profile'] = kwargs[key]
                elif key == 'version':
                    self.parameters.setdefault('properties', {})['version'] = kwargs[key]
                elif key == 'enforce_ssl':
                    self.parameters.setdefault('properties', {})['ssl_enforcement'] = 'Enabled' if kwargs[key] else 'Disabled'
                elif key == 'create_mode':
                    self.parameters.setdefault('properties', {})['create_mode'] = kwargs[key]
                elif key == 'admin_username':
                    self.parameters.setdefault('properties', {})['administrator_login'] = kwargs[key]
                elif key == 'admin_password':
                    self.parameters.setdefault('properties', {})['administrator_login_password'] = kwargs[key]
        old_response = None
        response = None
        resource_group = self.get_resource_group(self.resource_group)
        if 'location' not in self.parameters:
            self.parameters['location'] = resource_group.location
        old_response = self.get_mysqlserver()
        if not old_response:
            self.log("MySQL Server instance doesn't exist")
            if self.restarted:
                self.fail("Mysql server instance doesn't exist, can't be restart")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            if self.restarted:
                self.restart_mysqlserver()
                self.results['changed'] = True
                self.results['state'] = old_response
                return self.results
            self.log('MySQL Server instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            elif self.state == 'present':
                self.log('Need to check if MySQL Server instance has to be deleted or may be updated')
                update_tags, newtags = self.update_tags(old_response.get('tags', {}))
                if update_tags:
                    self.tags = newtags
                self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the MySQL Server instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_mysqlserver()
            if not old_response:
                self.results['changed'] = True
            else:
                self.results['changed'] = old_response.__ne__(response)
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('MySQL Server instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_mysqlserver()
            while self.get_mysqlserver():
                time.sleep(20)
        else:
            self.log('MySQL Server instance unchanged')
            self.results['changed'] = False
            response = old_response
        if response:
            self.results['id'] = response['id']
            self.results['version'] = response['version']
            self.results['state'] = response['user_visible_state']
            self.results['fully_qualified_domain_name'] = response['fully_qualified_domain_name']
        return self.results

    def restart_mysqlserver(self):
        """
        Restart MySQL Server.
        """
        self.log('Restart MySQL Server instance {0}'.format(self.name))
        try:
            response = self.mysql_client.servers.begin_restart(resource_group_name=self.resource_group, server_name=self.name)
        except Exception as exc:
            self.fail('Error restarting mysql server {0} - {1}'.format(self.name, str(exc)))
        return True

    def create_update_mysqlserver(self):
        """
        Creates or updates MySQL Server with the specified configuration.

        :return: deserialized MySQL Server instance state dictionary
        """
        self.log('Creating / Updating the MySQL Server instance {0}'.format(self.name))
        try:
            self.parameters['tags'] = self.tags
            if self.to_do == Actions.Create:
                response = self.mysql_client.servers.begin_create(resource_group_name=self.resource_group, server_name=self.name, parameters=self.parameters)
            else:
                self.parameters.update(self.parameters.pop('properties', {}))
                response = self.mysql_client.servers.begin_update(resource_group_name=self.resource_group, server_name=self.name, parameters=self.parameters)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the MySQL Server instance.')
            self.fail('Error creating the MySQL Server instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_mysqlserver(self):
        """
        Deletes specified MySQL Server instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the MySQL Server instance {0}'.format(self.name))
        try:
            response = self.mysql_client.servers.begin_delete(resource_group_name=self.resource_group, server_name=self.name)
        except Exception as e:
            self.log('Error attempting to delete the MySQL Server instance.')
            self.fail('Error deleting the MySQL Server instance: {0}'.format(str(e)))
        return True

    def get_mysqlserver(self):
        """
        Gets the properties of the specified MySQL Server.

        :return: deserialized MySQL Server instance state dictionary
        """
        self.log('Checking if the MySQL Server instance {0} is present'.format(self.name))
        found = False
        try:
            response = self.mysql_client.servers.get(resource_group_name=self.resource_group, server_name=self.name)
            found = True
            self.log('Response : {0}'.format(response))
            self.log('MySQL Server instance : {0} found'.format(response.name))
        except ResourceNotFoundError as e:
            self.log('Did not find the MySQL Server instance.')
        if found is True:
            return response.as_dict()
        return False