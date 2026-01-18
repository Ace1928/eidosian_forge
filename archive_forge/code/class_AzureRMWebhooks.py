from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMWebhooks(AzureRMModuleBase):
    """Configuration class for an Azure RM Webhook resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), registry_name=dict(type='str', required=True), webhook_name=dict(type='str', required=True), location=dict(type='str'), service_uri=dict(type='str'), custom_headers=dict(type='dict'), status=dict(type='str', choices=['enabled', 'disabled']), scope=dict(type='str'), actions=dict(type='list', elements='str'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.registry_name = None
        self.webhook_name = None
        self.parameters = dict()
        self.results = dict(changed=False)
        self.state = None
        self.to_do = Actions.NoAction
        super(AzureRMWebhooks, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                if key == 'location':
                    self.parameters['location'] = kwargs[key]
                elif key == 'service_uri':
                    self.parameters['service_uri'] = kwargs[key]
                elif key == 'custom_headers':
                    self.parameters['custom_headers'] = kwargs[key]
                elif key == 'status':
                    self.parameters['status'] = kwargs[key]
                elif key == 'scope':
                    self.parameters['scope'] = kwargs[key]
                elif key == 'actions':
                    self.parameters['actions'] = kwargs[key]
        old_response = None
        response = None
        resource_group = self.get_resource_group(self.resource_group)
        if 'location' not in self.parameters:
            self.parameters['location'] = resource_group.location
        old_response = self.get_webhook()
        if not old_response:
            self.log("Webhook instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('Webhook instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            elif self.state == 'present':
                self.log('Need to check if Webhook instance has to be deleted or may be updated')
                self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the Webhook instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_webhook()
            if not old_response:
                self.results['changed'] = True
            else:
                self.results['changed'] = old_response.__ne__(response)
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('Webhook instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_webhook()
            while self.get_webhook():
                time.sleep(20)
        else:
            self.log('Webhook instance unchanged')
            self.results['changed'] = False
            response = old_response
        if response:
            self.results['id'] = response['id']
            self.results['status'] = response['status']
        return self.results

    def create_update_webhook(self):
        """
        Creates or updates Webhook with the specified configuration.

        :return: deserialized Webhook instance state dictionary
        """
        self.log('Creating / Updating the Webhook instance {0}'.format(self.webhook_name))
        try:
            if self.to_do == Actions.Create:
                response = self.containerregistry_client.webhooks.begin_create(resource_group_name=self.resource_group, registry_name=self.registry_name, webhook_name=self.webhook_name, webhook_create_parameters=self.parameters)
            else:
                response = self.containerregistry_client.webhooks.begin_update(resource_group_name=self.resource_group, registry_name=self.registry_name, webhook_name=self.webhook_name, webhook_update_parameters=self.parameters)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the Webhook instance.')
            self.fail('Error creating the Webhook instance: {0}'.format(str(exc)))
        return create_webhook_dict(response)

    def delete_webhook(self):
        """
        Deletes specified Webhook instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the Webhook instance {0}'.format(self.webhook_name))
        try:
            response = self.containerregistry_client.webhooks.begin_delete(resource_group_name=self.resource_group, registry_name=self.registry_name, webhook_name=self.webhook_name)
            self.get_poller_result(response)
        except Exception as e:
            self.log('Error attempting to delete the Webhook instance.')
            self.fail('Error deleting the Webhook instance: {0}'.format(str(e)))
        return True

    def get_webhook(self):
        """
        Gets the properties of the specified Webhook.

        :return: deserialized Webhook instance state dictionary
        """
        self.log('Checking if the Webhook instance {0} is present'.format(self.webhook_name))
        found = False
        try:
            response = self.containerregistry_client.webhooks.get(resource_group_name=self.resource_group, registry_name=self.registry_name, webhook_name=self.webhook_name)
            found = True
            self.log('Response : {0}'.format(response))
            self.log('Webhook instance : {0} found'.format(response.name))
        except ResourceNotFoundError as e:
            self.log('Did not find the Webhook instance: {0}'.format(str(e)))
        if found is True:
            return response.as_dict()
        return False