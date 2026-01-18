from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _camel_to_snake
import re
class AzureRMTrafficManagerProfileInfo(AzureRMModuleBase):
    """Utility class to get Azure Traffic Manager profile facts"""

    def __init__(self):
        self.module_args = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False, tms=[])
        self.name = None
        self.resource_group = None
        self.tags = None
        super(AzureRMTrafficManagerProfileInfo, self).__init__(derived_arg_spec=self.module_args, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_trafficmanagerprofile_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_trafficmanagerprofile_facts' module has been renamed to 'azure_rm_trafficmanagerprofile_info'", version=(2.9,))
        for key in self.module_args:
            setattr(self, key, kwargs[key])
        if self.name and (not self.resource_group):
            self.fail('Parameter error: resource group required when filtering by name.')
        if self.name:
            self.results['tms'] = self.get_item()
        elif self.resource_group:
            self.results['tms'] = self.list_resource_group()
        else:
            self.results['tms'] = self.list_all()
        return self.results

    def get_item(self):
        """Get a single Azure Traffic Manager profile"""
        self.log('Get properties for {0}'.format(self.name))
        item = None
        result = []
        try:
            item = self.traffic_manager_management_client.profiles.get(self.resource_group, self.name)
        except ResourceNotFoundError:
            pass
        if item and self.has_tags(item.tags, self.tags):
            result = [self.serialize_tm(item)]
        return result

    def list_resource_group(self):
        """Get all Azure Traffic Managers profiles within a resource group"""
        self.log('List all Azure Traffic Managers within a resource group')
        try:
            response = self.traffic_manager_management_client.profiles.list_by_resource_group(self.resource_group)
        except Exception as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(self.serialize_tm(item))
        return results

    def list_all(self):
        """Get all Azure Traffic Manager profiles within a subscription"""
        self.log('List all Traffic Manager profiles within a subscription')
        try:
            response = self.traffic_manager_management_client.profiles.list_by_subscription()
        except Exception as exc:
            self.fail('Error listing all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(self.serialize_tm(item))
        return results

    def serialize_tm(self, tm):
        """
        Convert a Traffic Manager profile object to dict.
        :param tm: Traffic Manager profile object
        :return: dict
        """
        result = self.serialize_obj(tm, AZURE_OBJECT_CLASS)
        new_result = {}
        new_result['id'] = tm.id
        new_result['resource_group'] = re.sub('\\/.*', '', re.sub('.*resourceGroups\\/', '', result['id']))
        new_result['name'] = tm.name
        new_result['state'] = 'present'
        new_result['location'] = tm.location
        new_result['profile_status'] = tm.profile_status
        new_result['routing_method'] = tm.traffic_routing_method.lower()
        new_result['dns_config'] = dict(relative_name=tm.dns_config.relative_name, fqdn=tm.dns_config.fqdn, ttl=tm.dns_config.ttl)
        new_result['monitor_config'] = dict(profile_monitor_status=tm.monitor_config.profile_monitor_status, protocol=tm.monitor_config.protocol, port=tm.monitor_config.port, path=tm.monitor_config.path, interval=tm.monitor_config.interval_in_seconds, timeout=tm.monitor_config.timeout_in_seconds, tolerated_failures=tm.monitor_config.tolerated_number_of_failures)
        new_result['endpoints'] = [serialize_endpoint(endpoint) for endpoint in tm.endpoints]
        new_result['tags'] = tm.tags
        return new_result