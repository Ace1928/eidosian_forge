from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
class AzureRMIPGroup(AzureRMModuleBase):

    def __init__(self):
        _load_params()
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), ip_addresses=dict(type='list', elements='str'), state=dict(choices=['present', 'absent'], default='present', type='str'))
        self.results = dict(changed=False, state=dict())
        self.resource_group = None
        self.name = None
        self.state = None
        self.location = None
        self.ip_addresses = None
        self.tags = None
        super(AzureRMIPGroup, self).__init__(self.module_arg_spec, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        changed = False
        results = dict()
        ip_group_old = None
        ip_group_new = None
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        self.location = normalize_location_name(self.location)
        try:
            self.log('Fetching IP group {0}'.format(self.name))
            ip_group_old = self.network_client.ip_groups.get(self.resource_group, self.name)
            results = self.ipgroup_to_dict(ip_group_old)
            if self.state == 'present':
                changed = False
                update_tags, results['tags'] = self.update_tags(results['tags'])
                if update_tags:
                    changed = True
                self.tags = results['tags']
                update_ip_address = self.ip_addresses_changed(self.ip_addresses, results['ip_addresses'])
                if update_ip_address:
                    changed = True
                    results['ip_addresses'] = self.ip_addresses
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
                ip_group_new = self.network_models.IpGroup(location=self.location, ip_addresses=self.ip_addresses)
                if self.tags:
                    ip_group_new.tags = self.tags
                self.results['state'] = self.create_or_update_ipgroup(ip_group_new)
            elif self.state == 'absent':
                self.delete_ipgroup()
                self.results['state'] = 'Deleted'
        return self.results

    def create_or_update_ipgroup(self, ip_group):
        try:
            response = self.network_client.ip_groups.begin_create_or_update(resource_group_name=self.resource_group, ip_groups_name=self.name, parameters=ip_group)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.fail('Error creating or updating IP group {0} - {1}'.format(self.name, str(exc)))
        return self.ipgroup_to_dict(response)

    def delete_ipgroup(self):
        try:
            response = self.network_client.ip_groups.begin_delete(resource_group_name=self.resource_group, ip_groups_name=self.name)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.fail('Error deleting IP group {0} - {1}'.format(self.name, str(exc)))
        return response

    def ip_addresses_changed(self, input_records, ip_group_records):
        input_set = set(input_records)
        ip_group_set = set(ip_group_records)
        changed = input_set != ip_group_set
        return changed

    def ipgroup_to_dict(self, ipgroup):
        result = ipgroup.as_dict()
        result['tags'] = ipgroup.tags
        return result