from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
class AzureRMTrafficManagerProfile(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), location=dict(type='str', default='global'), profile_status=dict(type='str', default='enabled', choices=['enabled', 'disabled']), routing_method=dict(type='str', default='performance', choices=['performance', 'priority', 'weighted', 'geographic']), dns_config=dict(type='dict', options=dns_config_spec), monitor_config=dict(type='dict', default=dict(protocol='HTTP', port=80, path='/'), options=monitor_config_spec))
        self.resource_group = None
        self.name = None
        self.state = None
        self.tags = None
        self.location = None
        self.profile_status = None
        self.routing_method = None
        self.dns_config = None
        self.monitor_config = None
        self.endpoints_copy = None
        self.results = dict(changed=False)
        super(AzureRMTrafficManagerProfile, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        to_be_updated = False
        if not self.dns_config:
            self.dns_config = dict(relative_name=self.name, ttl=60)
        if not self.location:
            self.location = 'global'
        response = self.get_traffic_manager_profile()
        if self.state == 'present':
            if not response:
                to_be_updated = True
            else:
                self.results = shorten_traffic_manager_dict(response)
                self.log('Results : {0}'.format(response))
                update_tags, response['tags'] = self.update_tags(response['tags'])
                if update_tags:
                    to_be_updated = True
                to_be_updated = to_be_updated or self.check_update(response)
            if to_be_updated:
                self.log('Need to Create / Update the Traffic Manager profile')
                if not self.check_mode:
                    self.results = shorten_traffic_manager_dict(self.create_update_traffic_manager_profile())
                    self.log('Creation / Update done.')
                self.results['changed'] = True
                return self.results
        elif self.state == 'absent' and response:
            self.log('Need to delete the Traffic Manager profile')
            self.results = shorten_traffic_manager_dict(response)
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_traffic_manager_profile()
            self.log('Traffic Manager profile deleted')
        return self.results

    def get_traffic_manager_profile(self):
        """
        Gets the properties of the specified Traffic Manager profile

        :return: deserialized Traffic Manager profile dict
        """
        self.log('Checking if Traffic Manager profile {0} is present'.format(self.name))
        try:
            response = self.traffic_manager_management_client.profiles.get(self.resource_group, self.name)
            self.log('Response : {0}'.format(response))
            self.log('Traffic Manager profile : {0} found'.format(response.name))
            self.endpoints_copy = response.endpoints if response and response.endpoints else None
            return traffic_manager_profile_to_dict(response)
        except ResourceNotFoundError:
            self.log('Did not find the Traffic Manager profile.')
            return False

    def delete_traffic_manager_profile(self):
        """
        Deletes the specified Traffic Manager profile in the specified subscription and resource group.
        :return: True
        """
        self.log('Deleting the Traffic Manager profile {0}'.format(self.name))
        try:
            operation_result = self.traffic_manager_management_client.profiles.delete(self.resource_group, self.name)
            return True
        except Exception as e:
            self.log('Error attempting to delete the Traffic Manager profile.')
            self.fail('Error deleting the Traffic Manager profile: {0}'.format(e.message))
            return False

    def create_update_traffic_manager_profile(self):
        """
        Creates or updates a Traffic Manager profile.

        :return: deserialized Traffic Manager profile state dictionary
        """
        self.log('Creating / Updating the Traffic Manager profile {0}'.format(self.name))
        parameters = Profile(tags=self.tags, location=self.location, profile_status=self.profile_status, traffic_routing_method=self.routing_method, dns_config=create_dns_config_instance(self.dns_config) if self.dns_config else None, monitor_config=create_monitor_config_instance(self.monitor_config) if self.monitor_config else None, endpoints=self.endpoints_copy)
        try:
            response = self.traffic_manager_management_client.profiles.create_or_update(self.resource_group, self.name, parameters)
            return traffic_manager_profile_to_dict(response)
        except Exception as exc:
            self.log('Error attempting to create the Traffic Manager.')
            self.fail('Error creating the Traffic Manager: {0}'.format(exc.message))

    def check_update(self, response):
        if self.location and normalize_location_name(response['location']) != normalize_location_name(self.location):
            self.log('Location Diff - Origin {0} / Update {1}'.format(response['location'], self.location))
            return True
        if self.profile_status and response['profile_status'].lower() != self.profile_status:
            self.log('Profile Status Diff - Origin {0} / Update {1}'.format(response['profile_status'], self.profile_status))
            return True
        if self.routing_method and response['routing_method'].lower() != self.routing_method:
            self.log('Traffic Routing Method Diff - Origin {0} / Update {1}'.format(response['routing_method'], self.routing_method))
            return True
        if self.dns_config and (response['dns_config']['relative_name'] != self.dns_config['relative_name'] or response['dns_config']['ttl'] != self.dns_config['ttl']):
            self.log('DNS Config Diff - Origin {0} / Update {1}'.format(response['dns_config'], self.dns_config))
            return True
        for k, v in self.monitor_config.items():
            if v:
                if str(v).lower() != str(response['monitor_config'][k]).lower():
                    self.log('Monitor Config Diff - Origin {0} / Update {1}'.format(response['monitor_config'], self.monitor_config))
                    return True
        return False