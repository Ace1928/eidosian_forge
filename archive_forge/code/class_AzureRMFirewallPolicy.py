from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, \
import copy
class AzureRMFirewallPolicy(AzureRMModuleBase):

    def __init__(self):
        _load_params()
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), base_policy=dict(type='str'), threat_intel_mode=dict(choices=['alert', 'deny', 'off'], default='alert', type='str'), threat_intel_whitelist=dict(type='dict', options=threat_intel_whitelist_spec), state=dict(choices=['present', 'absent'], default='present', type='str'))
        self.results = dict(changed=False, state=dict())
        self.resource_group = None
        self.name = None
        self.state = None
        self.location = None
        self.base_policy = None
        self.threat_intel_mode = None
        self.threat_intel_whitelist = None
        self.tags = None
        super(AzureRMFirewallPolicy, self).__init__(self.module_arg_spec, supports_tags=True, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        changed = False
        results = dict()
        firewall_policy_old = None
        firewall_policy_new = None
        update_ip_address = False
        update_fqdns = False
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        self.location = normalize_location_name(self.location)
        if self.base_policy:
            base_policy = self.parse_resource_to_dict(self.base_policy)
            self.base_policy = format_resource_id(val=base_policy['name'], subscription_id=base_policy['subscription_id'], namespace='Microsoft.Network', types='firewallPolicies', resource_group=base_policy['resource_group'])
        try:
            self.log('Fetching Firewall policy {0}'.format(self.name))
            firewall_policy_old = self.network_client.firewall_policies.get(self.resource_group, self.name)
            results = self.firewallpolicy_to_dict(firewall_policy_old)
            if self.state == 'present':
                changed = False
                update_tags, results['tags'] = self.update_tags(results['tags'])
                if update_tags:
                    changed = True
                self.tags = results['tags']
                if self.base_policy is not None:
                    if 'base_policy' not in results and self.base_policy != '' or ('base_policy' in results and self.base_policy != results['base_policy']['id']):
                        changed = True
                        results['base_policy'] = self.base_policy
                if self.threat_intel_mode is not None and self.threat_intel_mode.lower() != results['threat_intel_mode'].lower():
                    changed = True
                    results['threat_intel_mode'] = self.threat_intel_mode
                if self.threat_intel_whitelist is not None:
                    if 'threat_intel_whitelist' not in results:
                        changed = True
                        results['threat_intel_whitelist'] = self.threat_intel_whitelist
                    else:
                        update_ip_addresses, results['threat_intel_whitelist']['ip_addresses'] = self.update_values(results['threat_intel_whitelist']['ip_addresses'] if 'ip_addresses' in results['threat_intel_whitelist'] else [], self.threat_intel_whitelist['ip_addresses'] if self.threat_intel_whitelist['ip_addresses'] is not None else [], self.threat_intel_whitelist['append_ip_addresses'])
                        update_fqdns, results['threat_intel_whitelist']['fqdns'] = self.update_values(results['threat_intel_whitelist']['fqdns'] if 'fqdns' in results['threat_intel_whitelist'] else [], self.threat_intel_whitelist['fqdns'] if self.threat_intel_whitelist['fqdns'] is not None else [], self.threat_intel_whitelist['append_fqdns'])
                        if update_ip_addresses:
                            changed = True
                        self.threat_intel_whitelist['ip_addresses'] = results['threat_intel_whitelist']['ip_addresses']
                        if update_fqdns:
                            changed = True
                        self.threat_intel_whitelist['fqdns'] = results['threat_intel_whitelist']['fqdns']
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
                firewall_policy_new = self.network_models.FirewallPolicy(location=self.location, threat_intel_mode=self.threat_intel_mode)
                if self.base_policy:
                    firewall_policy_new.base_policy = self.network_models.FirewallPolicy(id=self.base_policy)
                if self.threat_intel_whitelist:
                    firewall_policy_new.threat_intel_whitelist = self.network_models.FirewallPolicyThreatIntelWhitelist(ip_addresses=self.threat_intel_whitelist['ip_addresses'], fqdns=self.threat_intel_whitelist['fqdns'])
                if self.tags:
                    firewall_policy_new.tags = self.tags
                self.results['state'] = self.create_or_update_firewallpolicy(firewall_policy_new)
            elif self.state == 'absent':
                self.delete_firewallpolicy()
                self.results['state'] = 'Deleted'
        return self.results

    def create_or_update_firewallpolicy(self, firewall_policy):
        try:
            response = self.network_client.firewall_policies.begin_create_or_update(resource_group_name=self.resource_group, firewall_policy_name=self.name, parameters=firewall_policy)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.fail('Error creating or updating Firewall policy {0} - {1}'.format(self.name, str(exc)))
        return self.firewallpolicy_to_dict(response)

    def delete_firewallpolicy(self):
        try:
            response = self.network_client.firewall_policies.begin_delete(resource_group_name=self.resource_group, firewall_policy_name=self.name)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.fail('Error deleting Firewall policy {0} - {1}'.format(self.name, str(exc)))
        return response

    def update_values(self, existing_values, param_values, append):
        new_values = copy.copy(existing_values)
        changed = False
        for item in param_values:
            if item not in new_values:
                changed = True
                new_values.append(item)
        if not append:
            for item in existing_values:
                if item not in param_values:
                    new_values.remove(item)
                    changed = True
        return (changed, new_values)

    def firewallpolicy_to_dict(self, firewallpolicy):
        result = firewallpolicy.as_dict()
        result['tags'] = firewallpolicy.tags
        return result