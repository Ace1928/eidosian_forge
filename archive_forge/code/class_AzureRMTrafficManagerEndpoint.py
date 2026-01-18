from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
class AzureRMTrafficManagerEndpoint(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), profile_name=dict(type='str', required=True), type=dict(type='str', choices=['azure_endpoints', 'external_endpoints', 'nested_endpoints'], required=True), target=dict(type='str'), target_resource_id=dict(type='str'), enabled=dict(type='bool', default=True), weight=dict(type='int'), priority=dict(type='int'), location=dict(type='str'), min_child_endpoints=dict(type='int'), geo_mapping=dict(type='list', elements='str'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.state = None
        self.profile_name = None
        self.type = None
        self.target_resource_id = None
        self.enabled = None
        self.weight = None
        self.priority = None
        self.location = None
        self.min_child_endpoints = None
        self.geo_mapping = None
        self.endpoint_status = 'Enabled'
        self.action = Actions.NoAction
        self.results = dict(changed=False)
        super(AzureRMTrafficManagerEndpoint, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        if self.type:
            self.type = _snake_to_camel(self.type)
        to_be_updated = False
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        if self.enabled is not None and self.enabled is False:
            self.endpoint_status = 'Disabled'
        response = self.get_traffic_manager_endpoint()
        if response:
            self.log('Results : {0}'.format(response))
            self.results['id'] = response['id']
            if self.state == 'present':
                to_be_update = self.check_update(response)
                if to_be_update:
                    self.action = Actions.CreateOrUpdate
            elif self.state == 'absent':
                self.action = Actions.Delete
        elif self.state == 'present':
            self.action = Actions.CreateOrUpdate
        elif self.state == 'absent':
            self.fail('Traffic Manager endpoint {0} not exists.'.format(self.name))
        if self.action == Actions.CreateOrUpdate:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.create_update_traffic_manager_endpoint()
            self.results['id'] = response['id']
        if self.action == Actions.Delete:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.delete_traffic_manager_endpoint()
        return self.results

    def get_traffic_manager_endpoint(self):
        """
        Gets the properties of the specified Traffic Manager endpoint

        :return: deserialized Traffic Manager endpoint dict
        """
        self.log('Checking if Traffic Manager endpoint {0} is present'.format(self.name))
        try:
            response = self.traffic_manager_management_client.endpoints.get(self.resource_group, self.profile_name, self.type, self.name)
            self.log('Response : {0}'.format(response))
            return traffic_manager_endpoint_to_dict(response)
        except ResourceNotFoundError:
            self.log('Did not find the Traffic Manager endpoint.')
            return False

    def delete_traffic_manager_endpoint(self):
        """
        Deletes the specified Traffic Manager endpoint.
        :return: True
        """
        self.log('Deleting the Traffic Manager endpoint {0}'.format(self.name))
        try:
            operation_result = self.traffic_manager_management_client.endpoints.delete(self.resource_group, self.profile_name, self.type, self.name)
            return True
        except Exception as exc:
            request_id = exc.request_id if exc.request_id else ''
            self.fail('Error deleting the Traffic Manager endpoint {0}, request id {1} - {2}'.format(self.name, request_id, str(exc)))
            return False

    def create_update_traffic_manager_endpoint(self):
        """
        Creates or updates a Traffic Manager endpoint.

        :return: deserialized Traffic Manager endpoint state dictionary
        """
        self.log('Creating / Updating the Traffic Manager endpoint {0}'.format(self.name))
        parameters = Endpoint(target_resource_id=self.target_resource_id, target=self.target, endpoint_status=self.endpoint_status, weight=self.weight, priority=self.priority, endpoint_location=self.location, min_child_endpoints=self.min_child_endpoints, geo_mapping=self.geo_mapping)
        try:
            response = self.traffic_manager_management_client.endpoints.create_or_update(self.resource_group, self.profile_name, self.type, self.name, parameters)
            return traffic_manager_endpoint_to_dict(response)
        except Exception as exc:
            request_id = exc.request_id if exc.request_id else ''
            self.fail('Error creating the Traffic Manager endpoint {0}, request id {1} - {2}'.format(self.name, request_id, str(exc)))

    def check_update(self, response):
        if self.endpoint_status is not None and response['status'].lower() != self.endpoint_status.lower():
            self.log('Status Diff - Origin {0} / Update {1}'.format(response['status'], self.endpoint_status))
            return True
        if self.type and response['type'].lower() != 'Microsoft.network/TrafficManagerProfiles/{0}'.format(self.type).lower():
            self.log('Type Diff - Origin {0} / Update {1}'.format(response['type'], self.type))
            return True
        if self.target_resource_id and response['target_resource_id'] != self.target_resource_id:
            self.log('target_resource_id Diff - Origin {0} / Update {1}'.format(response['target_resource_id'], self.target_resource_id))
            return True
        if self.target and response['target'] != self.target:
            self.log('target Diff - Origin {0} / Update {1}'.format(response['target'], self.target))
            return True
        if self.weight and int(response['weight']) != self.weight:
            self.log('weight Diff - Origin {0} / Update {1}'.format(response['weight'], self.weight))
            return True
        if self.priority and int(response['priority']) != self.priority:
            self.log('priority Diff - Origin {0} / Update {1}'.format(response['priority'], self.priority))
            return True
        if self.min_child_endpoints and int(response['min_child_endpoints']) != self.min_child_endpoints:
            self.log('min_child_endpoints Diff - Origin {0} / Update {1}'.format(response['min_child_endpoints'], self.min_child_endpoints))
            return True
        if self.geo_mapping and response['geo_mapping'] != self.geo_mapping:
            self.log('geo_mapping Diff - Origin {0} / Update {1}'.format(response['geo_mapping'], self.geo_mapping))
            return True
        return False