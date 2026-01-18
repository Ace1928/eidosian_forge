from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMFirewallPolicyInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.name = None
        self.resource_group = None
        self.tags = None
        super(AzureRMFirewallPolicyInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, facts_module=True, supports_tags=False)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        results = []
        if self.name is not None:
            results = self.get_item()
        elif self.resource_group:
            results = self.list_resource_group()
        else:
            results = self.list_items()
        self.results['firewallpolicies'] = self.curated_items(results)
        return self.results

    def get_item(self):
        self.log('Get properties for Firewall policy - {0}'.format(self.name))
        item = None
        results = []
        try:
            item = self.network_client.firewall_policies.get(self.resource_group, self.name)
        except ResourceNotFoundError:
            pass
        if item and self.has_tags(item.tags, self.tags):
            results = [item]
        return results

    def list_resource_group(self):
        self.log('List all Firewall policies for resource group - {0}'.format(self.resource_group))
        try:
            response = self.network_client.firewall_policies.list(self.resource_group)
        except Exception as exc:
            self.fail('Failed to list firewall policies for resource group {0} - {1}'.format(self.resource_group, str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(item)
        return results

    def list_items(self):
        self.log('List all the Firewall Policies in a subscription.')
        try:
            response = self.network_client.firewall_policies.list_all()
        except Exception as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(item)
        return results

    def curated_items(self, raws):
        return [self.firewallpolicy_to_dict(item) for item in raws] if raws else []

    def firewallpolicy_to_dict(self, firewallpolicy):
        result = dict(id=firewallpolicy.id, name=firewallpolicy.name, location=firewallpolicy.location, tags=firewallpolicy.tags, rule_collection_groups=[dict(id=x.id) for x in firewallpolicy.rule_collection_groups], provisioning_state=firewallpolicy.provisioning_state, base_policy=firewallpolicy.base_policy.id if firewallpolicy.base_policy is not None else None, firewalls=[dict(id=x.id) for x in firewallpolicy.firewalls], child_policies=[dict(id=x.id) for x in firewallpolicy.child_policies], threat_intel_mode=firewallpolicy.threat_intel_mode, threat_intel_whitelist=dict(ip_addresses=firewallpolicy.threat_intel_whitelist.ip_addresses, fqdns=firewallpolicy.threat_intel_whitelist.fqdns) if firewallpolicy.threat_intel_whitelist is not None else dict(), dns_settings=dict(enable_proxy=firewallpolicy.dns_settings.enable_proxy, servers=firewallpolicy.dns_settings.servers, require_proxy_for_network_rules=firewallpolicy.dns_settings.require_proxy_for_network_rules) if firewallpolicy.dns_settings is not None else dict(), etag=firewallpolicy.etag, type=firewallpolicy.type)
        return result