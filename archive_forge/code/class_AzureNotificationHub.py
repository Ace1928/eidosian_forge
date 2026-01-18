from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureNotificationHub(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), namespace_name=dict(type='str', required=True), name=dict(type='str'), location=dict(type='str'), sku=dict(type='str', choices=['free', 'basic', 'standard'], default='free'), state=dict(choices=['present', 'absent'], default='present', type='str'))
        self.resource_group = None
        self.namespace_name = None
        self.name = None
        self.sku = None
        self.location = None
        self.authorizations = None
        self.tags = None
        self.state = None
        self.results = dict(changed=False, state=dict())
        super(AzureNotificationHub, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        self.results['check_mode'] = self.check_mode
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        results = dict()
        changed = False
        try:
            self.log('Fetching Notification Hub Namespace {0}'.format(self.name))
            namespace = self.notification_hub_client.namespaces.get(self.resource_group, self.namespace_name)
            results = namespace_to_dict(namespace)
            if self.name:
                self.log('Fetching Notification Hub {0}'.format(self.name))
                notification_hub = self.notification_hub_client.notification_hubs.get(self.resource_group, self.namespace_name, self.name)
                results = notification_hub_to_dict(notification_hub)
            if self.state == 'present':
                changed = False
                update_tags, results['tags'] = self.update_tags(results['tags'])
                if update_tags:
                    changed = True
                elif self.namespace_name and (not self.name):
                    if self.sku != results['sku']['name'].lower():
                        changed = True
            elif self.state == 'absent':
                changed = True
        except ResourceNotFoundError:
            if self.state == 'present':
                changed = True
            else:
                changed = False
        self.results['changed'] = changed
        if self.name and (not changed):
            self.results['state'] = results
        if self.check_mode:
            return self.results
        if changed:
            if self.state == 'present':
                if self.name is None:
                    self.results['state'] = self.create_or_update_namespaces()
                elif self.namespace_name and self.name:
                    self.results['state'] = self.create_or_update_notification_hub()
            elif self.state == 'absent':
                if self.name is None:
                    self.delete_namespace()
                elif self.namespace_name and self.name:
                    self.delete_notification_hub()
                self.results['state']['status'] = 'Deleted'
        return self.results

    def create_or_update_namespaces(self):
        """
        create or update namespaces
        """
        try:
            namespace_params = NamespaceCreateOrUpdateParameters(location=self.location, namespace_type='NotificationHub', sku=Sku(name=self.sku), tags=self.tags)
            result = self.notification_hub_client.namespaces.create_or_update(self.resource_group, self.namespace_name, namespace_params)
            namespace = self.notification_hub_client.namespaces.get(self.resource_group, self.namespace_name)
            while namespace.status == 'Created':
                time.sleep(30)
                namespace = self.notification_hub_client.namespaces.get(self.resource_group, self.namespace_name)
        except Exception as ex:
            self.fail('Failed to create namespace {0} in resource group {1}: {2}'.format(self.namespace_name, self.resource_group, str(ex)))
        return namespace_to_dict(result)

    def create_or_update_notification_hub(self):
        """
        Create or update Notification Hub.
        :return: create or update Notification Hub instance state dictionary
        """
        try:
            response = self.create_or_update_namespaces()
            params = NotificationHubCreateOrUpdateParameters(location=self.location, sku=Sku(name=self.sku), tags=self.tags)
            result = self.notification_hub_client.notification_hubs.create_or_update(self.resource_group, self.namespace_name, self.name, params)
            self.log('Response : {0}'.format(result))
        except Exception as ex:
            self.fail('Failed to create notification hub {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))
        return notification_hub_to_dict(result)

    def delete_notification_hub(self):
        """
        Deletes specified notication hub
        :return True
        """
        self.log('Deleting the notification hub {0}'.format(self.name))
        try:
            result = self.notification_hub_client.notification_hubs.delete(self.resource_group, self.namespace_name, self.name)
        except Exception as e:
            self.log('Error attempting to delete notification hub.')
            self.fail('Error deleting the notification hub : {0}'.format(str(e)))
        return True

    def delete_namespace(self):
        """
        Deletes specified namespace
        :return True
        """
        self.log('Deleting the namespace {0}'.format(self.namespace_name))
        try:
            result = self.notification_hub_client.namespaces.begin_delete(self.resource_group, self.namespace_name)
        except Exception as e:
            self.log('Error attempting to delete namespace.')
            self.fail('Error deleting the namespace : {0}'.format(str(e)))
        return True