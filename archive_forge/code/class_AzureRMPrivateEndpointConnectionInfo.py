from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMPrivateEndpointConnectionInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), service_name=dict(type='str', required=True), resource_group=dict(type='str', required=True))
        self.name = None
        self.service_name = None
        self.resource_group = None
        self.results = dict(changed=False)
        super(AzureRMPrivateEndpointConnectionInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        if self.name is not None:
            self.results['endpoint_connection'] = self.get_item()
        else:
            self.results['endpoint_connection'] = self.list_items()
        return self.results

    def get_item(self):
        self.log('Get properties for {0} in {1}'.format(self.name, self.service_name))
        try:
            response = self.network_client.private_link_services.get_private_endpoint_connection(self.resource_group, self.service_name, self.name)
            return [self.connect_to_dict(response)]
        except ResourceNotFoundError:
            self.log('Could not get info for {0} in {1}'.format(self.name, self.service_name))
        return []

    def list_items(self):
        result = []
        self.log('List all in {0}'.format(self.service_name))
        try:
            response = self.network_client.private_link_services.list_private_endpoint_connections(self.resource_group, self.service_name)
            while True:
                result.append(response.next())
        except StopIteration:
            pass
        except Exception as exc:
            self.fail('Failed to list all items in {0}: {1}'.format(self.service_name, str(exc)))
        return [self.connect_to_dict(item) for item in result]

    def connect_to_dict(self, connect_info):
        connect = connect_info.as_dict()
        result = dict(id=connect.get('id'), name=connect.get('name'), type=connect.get('type'), etag=connect.get('etag'), private_endpoint=dict(), private_link_service_connection_state=dict(), provisioning_state=connect.get('provisioning_state'), link_identifier=connect.get('link_identifier'))
        if connect.get('private_endpoint') is not None:
            result['private_endpoint']['id'] = connect.get('private_endpoint')['id']
        if connect.get('private_link_service_connection_state') is not None:
            result['private_link_service_connection_state']['status'] = connect.get('private_link_service_connection_state')['status']
            result['private_link_service_connection_state']['description'] = connect.get('private_link_service_connection_state')['description']
            result['private_link_service_connection_state']['actions_required'] = connect.get('private_link_service_connection_state')['actions_required']
        return result