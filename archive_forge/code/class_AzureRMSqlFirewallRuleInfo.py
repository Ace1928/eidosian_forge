from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMSqlFirewallRuleInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), server_name=dict(type='str', required=True), name=dict(type='str'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.server_name = None
        self.name = None
        super(AzureRMSqlFirewallRuleInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_sqlfirewallrule_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_sqlfirewallrule_facts' module has been renamed to 'azure_rm_sqlfirewallrule_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name is not None:
            self.results['rules'] = self.get()
        else:
            self.results['rules'] = self.list_by_server()
        return self.results

    def get(self):
        """
        Gets facts of the specified SQL Firewall Rule.

        :return: deserialized SQL Firewall Ruleinstance state dictionary
        """
        response = None
        results = []
        try:
            response = self.sql_client.firewall_rules.get(resource_group_name=self.resource_group, server_name=self.server_name, firewall_rule_name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError:
            self.log('Could not get facts for FirewallRules.')
        if response is not None:
            results.append(self.format_item(response))
        return results

    def list_by_server(self):
        """
        Gets facts of the specified SQL Firewall Rule.

        :return: deserialized SQL Firewall Ruleinstance state dictionary
        """
        response = None
        results = []
        try:
            response = self.sql_client.firewall_rules.list_by_server(resource_group_name=self.resource_group, server_name=self.server_name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError:
            self.log('Could not get facts for FirewallRules.')
        if response is not None:
            for item in response:
                results.append(self.format_item(item))
        return results

    def format_item(self, item):
        d = item.as_dict()
        d = {'id': d['id'], 'resource_group': self.resource_group, 'server_name': self.server_name, 'name': d['name'], 'start_ip_address': d['start_ip_address'], 'end_ip_address': d['end_ip_address']}
        return d