from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
class AzureRMSecurityGroup(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(default_rules=dict(type='list', elements='dict', options=rule_spec, mutually_exclusive=[('source_application_security_groups', 'source_address_prefix'), ('destination_application_security_groups', 'destination_address_prefix')]), location=dict(type='str'), name=dict(type='str', required=True), purge_default_rules=dict(type='bool', default=False), purge_rules=dict(type='bool', default=False), resource_group=dict(required=True, type='str'), rules=dict(type='list', elements='dict', options=rule_spec, mutually_exclusive=[('source_application_security_groups', 'source_address_prefix'), ('destination_application_security_groups', 'destination_address_prefix')]), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.default_rules = None
        self.location = None
        self.name = None
        self.purge_default_rules = None
        self.purge_rules = None
        self.resource_group = None
        self.rules = None
        self.state = None
        self.tags = None
        self.nsg_models = None
        self.results = dict(changed=False, state=dict())
        super(AzureRMSecurityGroup, self).__init__(self.module_arg_spec, supports_check_mode=True)

    def exec_module(self, **kwargs):
        self.nsg_models = self.network_client.network_security_groups.models
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        changed = False
        results = dict()
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        if self.rules:
            for rule in self.rules:
                try:
                    validate_rule(self, rule)
                except Exception as exc:
                    self.fail('Error validating rule {0} - {1}'.format(rule, str(exc)))
                self.convert_asg_to_id(rule)
        if self.default_rules:
            for rule in self.default_rules:
                try:
                    validate_rule(self, rule, 'default')
                except Exception as exc:
                    self.fail('Error validating default rule {0} - {1}'.format(rule, str(exc)))
                self.convert_asg_to_id(rule)
        try:
            nsg = self.network_client.network_security_groups.get(self.resource_group, self.name)
            results = create_network_security_group_dict(nsg)
            self.log('Found security group:')
            self.log(results, pretty_print=True)
            self.check_provisioning_state(nsg, self.state)
            if self.state == 'present':
                pass
            elif self.state == 'absent':
                self.log("CHANGED: security group found but state is 'absent'")
                changed = True
        except ResourceNotFoundError:
            if self.state == 'present':
                self.log("CHANGED: security group not found and state is 'present'")
                changed = True
        if self.state == 'present' and (not changed):
            self.log('Update security group {0}'.format(self.name))
            update_tags, results['tags'] = self.update_tags(results['tags'])
            if update_tags:
                changed = True
            rule_changed, new_rule = compare_rules_change(results['rules'], self.rules, self.purge_rules)
            if rule_changed:
                changed = True
                results['rules'] = new_rule
            rule_changed, new_rule = compare_rules_change(results['default_rules'], self.default_rules, self.purge_default_rules)
            if rule_changed:
                changed = True
                results['default_rules'] = new_rule
            self.results['changed'] = changed
            self.results['state'] = results
            if not self.check_mode and changed:
                self.results['state'] = self.create_or_update(results)
        elif self.state == 'present' and changed:
            self.log('Create security group {0}'.format(self.name))
            if not self.location:
                self.fail('Parameter error: location required when creating a security group.')
            results['name'] = self.name
            results['location'] = self.location
            results['rules'] = []
            results['default_rules'] = []
            results['tags'] = {}
            if self.rules:
                results['rules'] = self.rules
            if self.default_rules:
                results['default_rules'] = self.default_rules
            if self.tags:
                results['tags'] = self.tags
            self.results['changed'] = changed
            self.results['state'] = results
            if not self.check_mode:
                self.results['state'] = self.create_or_update(results)
        elif self.state == 'absent' and changed:
            self.log('Delete security group {0}'.format(self.name))
            self.results['changed'] = changed
            self.results['state'] = dict()
            if not self.check_mode:
                self.delete()
                self.results['state']['status'] = 'Deleted'
        return self.results

    def create_or_update(self, results):
        parameters = self.nsg_models.NetworkSecurityGroup()
        if results.get('rules'):
            parameters.security_rules = []
            for rule in results.get('rules'):
                parameters.security_rules.append(create_rule_instance(self, rule))
        if results.get('default_rules'):
            parameters.default_security_rules = []
            for rule in results.get('default_rules'):
                parameters.default_security_rules.append(create_rule_instance(self, rule))
        parameters.tags = results.get('tags')
        parameters.location = results.get('location')
        try:
            poller = self.network_client.network_security_groups.begin_create_or_update(resource_group_name=self.resource_group, network_security_group_name=self.name, parameters=parameters)
            result = self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error creating/updating security group {0} - {1}'.format(self.name, str(exc)))
        return create_network_security_group_dict(result)

    def delete(self):
        try:
            poller = self.network_client.network_security_groups.begin_delete(resource_group_name=self.resource_group, network_security_group_name=self.name)
            result = self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error deleting security group {0} - {1}'.format(self.name, str(exc)))
        return result

    def convert_asg_to_id(self, rule):

        def convert_to_id(rule, key):
            if rule.get(key):
                ids = []
                for p in rule.get(key):
                    if isinstance(p, dict):
                        ids.append('/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/applicationSecurityGroups/{2}'.format(self.subscription_id, p.get('resource_group'), p.get('name')))
                    elif isinstance(p, str):
                        if is_valid_resource_id(p):
                            ids.append(p)
                        else:
                            ids.append('/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/applicationSecurityGroups/{2}'.format(self.subscription_id, self.resource_group, p))
                rule[key] = ids
        convert_to_id(rule, 'source_application_security_groups')
        convert_to_id(rule, 'destination_application_security_groups')