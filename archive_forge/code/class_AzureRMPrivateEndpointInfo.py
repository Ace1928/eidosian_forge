from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMPrivateEndpointInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False, privateendpoints=[])
        self.name = None
        self.resource_group = None
        self.tags = None
        self.results = dict(changed=False)
        super(AzureRMPrivateEndpointInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        if self.name is not None:
            self.results['privateendpoints'] = self.get_item()
        elif self.resource_group is not None:
            self.results['privateendpoints'] = self.list_resource_group()
        else:
            self.results['privateendpoints'] = self.list_items()
        return self.results

    def get_item(self):
        self.log('Get properties for {0}'.format(self.name))
        item = None
        results = []
        try:
            item = self.network_client.private_endpoints.get(self.resource_group, self.name)
        except ResourceNotFoundError:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        format_item = self.privateendpoints_to_dict(item)
        if format_item and self.has_tags(format_item['tags'], self.tags):
            results = [format_item]
        return results

    def list_resource_group(self):
        self.log('List items for resource group')
        try:
            response = self.network_client.private_endpoints.list(self.resource_group)
        except ResourceNotFoundError as exc:
            self.fail('Failed to list for resource group {0} - {1}'.format(self.resource_group, str(exc)))
        results = []
        for item in response:
            format_item = self.privateendpoints_to_dict(item)
            if self.has_tags(format_item['tags'], self.tags):
                results.append(format_item)
        return results

    def list_items(self):
        self.log('List all for items')
        try:
            response = self.network_client.private_endpoints.list_by_subscription()
        except ResourceNotFoundError as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            format_item = self.privateendpoints_to_dict(item)
            if self.has_tags(format_item['tags'], self.tags):
                results.append(format_item)
        return results

    def privateendpoints_to_dict(self, privateendpoint):
        if privateendpoint is None:
            return None
        results = dict(id=privateendpoint.id, name=privateendpoint.name, location=privateendpoint.location, tags=privateendpoint.tags, provisioning_state=privateendpoint.provisioning_state, type=privateendpoint.type, etag=privateendpoint.etag, subnet_id=privateendpoint.subnet.id)
        if privateendpoint.network_interfaces and len(privateendpoint.network_interfaces) > 0:
            results['network_interfaces'] = []
            for interface in privateendpoint.network_interfaces:
                results['network_interfaces'].append(interface.id)
        if privateendpoint.private_link_service_connections and len(privateendpoint.private_link_service_connections) > 0:
            results['private_link_service_connections'] = []
            for connections in privateendpoint.private_link_service_connections:
                connection = {}
                connection['connection_state'] = {}
                connection['id'] = connections.id
                connection['name'] = connections.name
                connection['type'] = connections.type
                connection['group_ids'] = connections.group_ids
                connection['connection_state']['status'] = connections.private_link_service_connection_state.status
                connection['connection_state']['description'] = connections.private_link_service_connection_state.description
                connection['connection_state']['actions_required'] = connections.private_link_service_connection_state.actions_required
                results['private_link_service_connections'].append(connection)
        if privateendpoint.manual_private_link_service_connections and len(privateendpoint.manual_private_link_service_connections) > 0:
            results['manual_private_link_service_connections'] = []
            for connections in privateendpoint.manual_private_link_service_connections:
                results['manual_private_link_service_connections'].append(connections.id)
        return results