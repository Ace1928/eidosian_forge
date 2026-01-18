from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
class AzureRMPrivateEndpoint(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), location=dict(type='str'), subnet=dict(type='dict', options=subnet_spec), private_link_service_connections=dict(type='list', elements='dict', options=private_service_connection_spec))
        self.resource_group = None
        self.name = None
        self.state = None
        self.location = None
        self.body = {}
        self.tags = None
        self.results = dict(changed=False, state=dict())
        self.to_do = Actions.NoAction
        super(AzureRMPrivateEndpoint, self).__init__(self.module_arg_spec, supports_tags=True, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.body[key] = kwargs[key]
        self.inflate_parameters(self.module_arg_spec, self.body, 0)
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        self.body['location'] = self.location
        self.body['tags'] = self.tags
        self.log('Fetching private endpoint {0}'.format(self.name))
        old_response = self.get_resource()
        if old_response is None:
            if self.state == 'present':
                self.to_do = Actions.Create
        elif self.state == 'absent':
            self.to_do = Actions.Delete
        else:
            update_tags, newtags = self.update_tags(old_response.get('tags', {}))
            if update_tags:
                self.body['tags'] = newtags
                self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.create_update_resource_private_endpoint(self.body)
        elif self.to_do == Actions.Delete:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.delete_private_endpoint()
        else:
            self.results['changed'] = False
            response = old_response
        if response is not None:
            self.results['state'] = response
        return self.results

    def create_update_resource_private_endpoint(self, privateendpoint):
        try:
            poller = self.network_client.private_endpoints.begin_create_or_update(resource_group_name=self.resource_group, private_endpoint_name=self.name, parameters=privateendpoint)
            new_privateendpoint = self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error creating or updating private endpoint {0} - {1}'.format(self.name, str(exc)))
        return self.private_endpoints_to_dict(new_privateendpoint)

    def delete_private_endpoint(self):
        try:
            poller = self.network_client.private_endpoints.begin_delete(self.resource_group, self.name)
            result = self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error deleting private endpoint {0} - {1}'.format(self.name, str(exc)))
        return result

    def get_resource(self):
        found = False
        try:
            private_endpoint = self.network_client.private_endpoints.get(self.resource_group, self.name)
            results = self.private_endpoints_to_dict(private_endpoint)
            found = True
            self.log('Response : {0}'.format(results))
        except ResourceNotFoundError:
            self.log('Did not find the private endpoint resource')
        if found is True:
            return results
        else:
            return None

    def private_endpoints_to_dict(self, privateendpoint):
        results = dict(id=privateendpoint.id, name=privateendpoint.name, location=privateendpoint.location, tags=privateendpoint.tags, provisioning_state=privateendpoint.provisioning_state, type=privateendpoint.type, etag=privateendpoint.etag, subnet=dict(id=privateendpoint.subnet.id))
        if privateendpoint.network_interfaces and len(privateendpoint.network_interfaces) > 0:
            results['network_interfaces'] = []
            for interface in privateendpoint.network_interfaces:
                results['network_interfaces'].append(interface.id)
        if privateendpoint.private_link_service_connections and len(privateendpoint.private_link_service_connections) > 0:
            results['private_link_service_connections'] = []
            for connections in privateendpoint.private_link_service_connections:
                results['private_link_service_connections'].append(dict(private_link_service_id=connections.private_link_service_id, name=connections.name))
        return results