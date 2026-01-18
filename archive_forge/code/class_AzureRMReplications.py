from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMReplications(AzureRMModuleBase):
    """Configuration class for an Azure RM Replication resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), registry_name=dict(type='str', required=True), replication_name=dict(type='str', required=True), replication=dict(type='dict'), location=dict(type='str'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.registry_name = None
        self.replication_name = None
        self.location = None
        self.results = dict(changed=False)
        self.state = None
        self.to_do = Actions.NoAction
        super(AzureRMReplications, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
        old_response = None
        response = None
        resource_group = self.get_resource_group(self.resource_group)
        if self.location is None:
            self.location = resource_group.location
        old_response = self.get_replication()
        if not old_response:
            self.log("Replication instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('Replication instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            elif self.state == 'present':
                self.log('Need to check if Replication instance has to be deleted or may be updated')
                self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the Replication instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_replication()
            if not old_response:
                self.results['changed'] = True
            else:
                self.results['changed'] = old_response.__ne__(response)
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('Replication instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_replication()
            while self.get_replication():
                time.sleep(20)
        else:
            self.log('Replication instance unchanged')
            self.results['changed'] = False
            response = old_response
        if response:
            self.results['id'] = response['id']
            self.results['status'] = response['status']
        return self.results

    def create_update_replication(self):
        """
        Creates or updates Replication with the specified configuration.

        :return: deserialized Replication instance state dictionary
        """
        self.log('Creating / Updating the Replication instance {0}'.format(self.replication_name))
        try:
            if self.to_do == Actions.Create:
                replication = Replication(location=self.location)
                response = self.containerregistry_client.replications.begin_create(resource_group_name=self.resource_group, registry_name=self.registry_name, replication_name=self.replication_name, replication=replication)
            else:
                update_params = ReplicationUpdateParameters()
                response = self.containerregistry_client.replications.begin_update(resource_group_name=self.resource_group, registry_name=self.registry_name, replication_name=self.replication_name, replication_update_parameters=update_params)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the Replication instance.')
            self.fail('Error creating the Replication instance: {0}'.format(str(exc)))
        return create_replication_dict(response)

    def delete_replication(self):
        """
        Deletes specified Replication instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the Replication instance {0}'.format(self.replication_name))
        try:
            response = self.containerregistry_client.replications.begin_delete(resource_group_name=self.resource_group, registry_name=self.registry_name, replication_name=self.replication_name)
            self.get_poller_result(response)
        except Exception as e:
            self.log('Error attempting to delete the Replication instance.')
            self.fail('Error deleting the Replication instance: {0}'.format(str(e)))
        return True

    def get_replication(self):
        """
        Gets the properties of the specified Replication.

        :return: deserialized Replication instance state dictionary
        """
        self.log('Checking if the Replication instance {0} is present'.format(self.replication_name))
        found = False
        try:
            response = self.containerregistry_client.replications.get(resource_group_name=self.resource_group, registry_name=self.registry_name, replication_name=self.replication_name)
            found = True
            self.log('Response : {0}'.format(response))
            self.log('Replication instance : {0} found'.format(response.name))
        except ResourceNotFoundError as e:
            self.log('Did not find the Replication instance: {0}'.format(str(e)))
        if found is True:
            return response.as_dict()
        return False