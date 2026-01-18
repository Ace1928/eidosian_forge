from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, \
class AzureRMHostGroup(AzureRMModuleBase):

    def __init__(self):
        _load_params()
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), platform_fault_domain_count=dict(type='int'), zones=dict(type='list', elements='str'), state=dict(choices=['present', 'absent'], default='present', type='str'))
        required_if = [('state', 'present', ['platform_fault_domain_count'])]
        self.results = dict(changed=False, state=dict())
        self.resource_group = None
        self.name = None
        self.state = None
        self.location = None
        self.tags = None
        self.platform_fault_domain_count = None
        self.zones = None
        super(AzureRMHostGroup, self).__init__(self.module_arg_spec, required_if=required_if, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        changed = False
        results = dict()
        host_group = None
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        self.location = normalize_location_name(self.location)
        try:
            self.log('Fetching host group {0}'.format(self.name))
            host_group = self.compute_client.dedicated_host_groups.get(self.resource_group, self.name)
            results = self.hostgroup_to_dict(host_group)
            if self.state == 'present':
                changed = False
                update_tags, results['tags'] = self.update_tags(results['tags'])
                if update_tags:
                    changed = True
                self.tags = results['tags']
                if self.platform_fault_domain_count != results['platform_fault_domain_count']:
                    self.fail('Error updating host group : {0}. Changing platform_fault_domain_count is not allowed.'.format(self.name))
                if self.zones:
                    if 'zones' in results and self.zones[0] != results['zones'][0] or 'zones' not in results:
                        self.fail('Error updating host group : {0}. Changing property zones is not allowed.'.format(self.name))
            elif self.state == 'absent':
                changed = True
        except ResourceNotFoundError:
            if self.state == 'present':
                changed = True
            else:
                changed = False
        self.results['changed'] = changed
        self.results['state'] = results
        if self.check_mode:
            return self.results
        if changed:
            if self.state == 'present':
                host_group_new = self.compute_models.DedicatedHostGroup(location=self.location, platform_fault_domain_count=self.platform_fault_domain_count, zones=self.zones)
                if self.tags:
                    host_group_new.tags = self.tags
                self.results['state'] = self.create_or_update_hostgroup(host_group_new)
            elif self.state == 'absent':
                self.delete_hostgroup()
                self.results['state'] = 'Deleted'
        return self.results

    def create_or_update_hostgroup(self, host_group):
        try:
            response = self.compute_client.dedicated_host_groups.create_or_update(resource_group_name=self.resource_group, host_group_name=self.name, parameters=host_group)
        except Exception as exc:
            self.fail('Error creating or updating host group {0} - {1}'.format(self.name, str(exc)))
        return self.hostgroup_to_dict(response)

    def delete_hostgroup(self):
        try:
            response = self.compute_client.dedicated_host_groups.delete(resource_group_name=self.resource_group, host_group_name=self.name)
        except Exception as exc:
            self.fail('Error deleting host group {0} - {1}'.format(self.name, str(exc)))
        return response

    def hostgroup_to_dict(self, hostgroup):
        result = hostgroup.as_dict()
        result['tags'] = hostgroup.tags
        return result