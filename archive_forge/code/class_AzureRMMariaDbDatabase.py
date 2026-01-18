from __future__ import absolute_import, division, print_function
import time
class AzureRMMariaDbDatabase(AzureRMModuleBase):
    """Configuration class for an Azure RM MariaDB Database resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), server_name=dict(type='str', required=True), name=dict(type='str', required=True), charset=dict(type='str'), collation=dict(type='str'), force_update=dict(type='bool', default=False), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.server_name = None
        self.name = None
        self.force_update = None
        self.parameters = dict()
        self.results = dict(changed=False)
        self.state = None
        self.to_do = Actions.NoAction
        super(AzureRMMariaDbDatabase, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                if key == 'charset':
                    self.parameters['charset'] = kwargs[key]
                elif key == 'collation':
                    self.parameters['collation'] = kwargs[key]
        old_response = None
        response = None
        resource_group = self.get_resource_group(self.resource_group)
        old_response = self.get_mariadbdatabase()
        if not old_response:
            self.log("MariaDB Database instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('MariaDB Database instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            elif self.state == 'present':
                self.log('Need to check if MariaDB Database instance has to be deleted or may be updated')
                if 'collation' in self.parameters and self.parameters['collation'] != old_response['collation']:
                    self.to_do = Actions.Update
                if 'charset' in self.parameters and self.parameters['charset'] != old_response['charset']:
                    self.to_do = Actions.Update
        if self.to_do == Actions.Update:
            if self.force_update:
                if not self.check_mode:
                    self.delete_mariadbdatabase()
            else:
                self.fail("Database properties cannot be updated without setting 'force_update' option")
                self.to_do = Actions.NoAction
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the MariaDB Database instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_mariadbdatabase()
            self.results['changed'] = True
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('MariaDB Database instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_mariadbdatabase()
            while self.get_mariadbdatabase():
                time.sleep(20)
        else:
            self.log('MariaDB Database instance unchanged')
            self.results['changed'] = False
            response = old_response
        if response:
            self.results['id'] = response['id']
            self.results['name'] = response['name']
        return self.results

    def create_update_mariadbdatabase(self):
        """
        Creates or updates MariaDB Database with the specified configuration.

        :return: deserialized MariaDB Database instance state dictionary
        """
        self.log('Creating / Updating the MariaDB Database instance {0}'.format(self.name))
        try:
            response = self.mariadb_client.databases.begin_create_or_update(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name, parameters=self.parameters)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the MariaDB Database instance.')
            self.fail('Error creating the MariaDB Database instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_mariadbdatabase(self):
        """
        Deletes specified MariaDB Database instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the MariaDB Database instance {0}'.format(self.name))
        try:
            response = self.mariadb_client.databases.begin_delete(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name)
        except Exception as e:
            self.log('Error attempting to delete the MariaDB Database instance.')
            self.fail('Error deleting the MariaDB Database instance: {0}'.format(str(e)))
        return True

    def get_mariadbdatabase(self):
        """
        Gets the properties of the specified MariaDB Database.

        :return: deserialized MariaDB Database instance state dictionary
        """
        self.log('Checking if the MariaDB Database instance {0} is present'.format(self.name))
        found = False
        try:
            response = self.mariadb_client.databases.get(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name)
            found = True
            self.log('Response : {0}'.format(response))
            self.log('MariaDB Database instance : {0} found'.format(response.name))
        except ResourceNotFoundError as e:
            self.log('Did not find the MariaDB Database instance.')
        if found is True:
            return response.as_dict()
        return False