from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureNotificationHubInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), namespace_name=dict(type='str'), name=dict(type='str'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.namespace_name = None
        self.name = None
        self.tags = None
        super(AzureNotificationHubInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name is None and self.namespace_name is None:
            results = self.list_all_namespace()
            self.results['namespaces'] = [self.namespace_to_dict(x) for x in results]
        elif self.name and self.namespace_name:
            results = self.get_notification_hub()
            self.results['notificationhub'] = [self.notification_hub_to_dict(x) for x in results]
        elif self.namespace_name:
            results = self.get_namespace()
            self.results['namespace'] = [self.namespace_to_dict(x) for x in results]
        return self.results

    def get_namespace(self):
        response = None
        results = []
        try:
            response = self.notification_hub_client.namespaces.get(self.resource_group, self.namespace_name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.fail('Could not get info for namespace. {0}').format(str(e))
        if response and self.has_tags(response.tags, self.tags):
            results = [response]
        return results

    def get_notification_hub(self):
        response = None
        results = []
        try:
            response = self.notification_hub_client.notification_hubs.get(self.resource_group, self.namespace_name, self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.fail('Could not get info for notification hub. {0}').format(str(e))
        if response and self.has_tags(response.tags, self.tags):
            results = [response]
        return results

    def list_all_namespace(self):
        self.log('List items for resource group')
        try:
            response = self.notification_hub_client.namespaces.list(self.resource_group)
        except Exception as exc:
            self.fail('Failed to list for resource group {0} - {1}'.format(self.resource_group, str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(item)
        return results

    def namespace_to_dict(self, item):
        namespace = item.as_dict()
        result = dict(additional_properties=namespace.get('additional_properties', {}), name=namespace.get('name', None), type=namespace.get('type', None), location=namespace.get('location', '').replace(' ', '').lower(), sku=namespace.get('sku'), tags=namespace.get('tags', None), provisioning_state=namespace.get('provisioning_state', None), region=namespace.get('region', None), metric_id=namespace.get('metric_id', None), service_bus_endpoint=namespace.get('service_bus_endpoint', None), scale_unit=namespace.get('scale_unit', None), enabled=namespace.get('enabled', None), critical=namespace.get('critical', None), data_center=namespace.get('data_center', None), namespace_type=namespace.get('namespace_type', None))
        return result

    def notification_hub_to_dict(self, item):
        notification_hub = item.as_dict()
        result = dict(additional_properties=notification_hub.get('additional_properties', {}), name=notification_hub.get('name', None), type=notification_hub.get('type', None), location=notification_hub.get('location', '').replace(' ', '').lower(), tags=notification_hub.get('tags', None), name_properties_name=notification_hub.get('name_properties_name', None), registration_ttl=notification_hub.get('registration_ttl', None), authorization_rules=notification_hub.get('authorization_rules', None), apns_credential=notification_hub.get('apns_credential', None), wns_credential=notification_hub.get('wns_credential', None), gcm_credential=notification_hub.get('gcm_credential', None), mpns_credential=notification_hub.get('mpns_credential', None), adm_credential=notification_hub.get('adm_credential', None), baidu_credential=notification_hub.get('baidu_credential', None))
        return result