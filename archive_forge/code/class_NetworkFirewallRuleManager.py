import time
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.arn import parse_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.base import BaseResourceManager
from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
from ansible_collections.community.aws.plugins.module_utils.base import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
class NetworkFirewallRuleManager(NFRuleGroupBoto3Mixin, BaseNetworkFirewallManager):
    RULE_TYPES = frozenset(['StatelessRulesAndCustomActions', 'StatefulRules', 'RulesSourceList', 'RulesString'])
    name = None
    rule_type = None
    arn = None

    def __init__(self, module, name=None, rule_type=None, arn=None):
        super().__init__(module)
        self.name = name
        self.rule_type = rule_type
        self.arn = arn
        if self.name or self.arn:
            rule_group = deepcopy(self.get_rule_group())
            self.original_resource = rule_group

    def _extra_error_output(self):
        output = super(NetworkFirewallRuleManager, self)._extra_error_output()
        if self.name:
            output['RuleGroupName'] = self.name
        if self.rule_type:
            output['Type'] = self.rule_type
        if self.arn:
            output['RuleGroupArn'] = self.arn
        return output

    def _filter_immutable_metadata_attributes(self, metadata):
        metadata = super(NetworkFirewallRuleManager, self)._filter_immutable_metadata_attributes(metadata)
        metadata.pop('RuleGroupArn', None)
        metadata.pop('RuleGroupName', None)
        metadata.pop('RuleGroupId', None)
        metadata.pop('Type', None)
        metadata.pop('Capacity', None)
        metadata.pop('RuleGroupStatus', None)
        metadata.pop('Tags', None)
        metadata.pop('ConsumedCapacity', None)
        metadata.pop('NumberOfAssociations', None)
        return metadata

    def _get_preupdate_arn(self):
        return self._get_metadata_value('RuleGroupArn')

    def _get_id_params(self, name=None, rule_type=None, arn=None):
        if arn:
            return dict(RuleGroupArn=arn)
        if self.arn:
            return dict(RuleGroupArn=self.arn)
        if not name:
            name = self.name
        if not rule_type:
            rule_type = self.rule_type
        if rule_type:
            rule_type = rule_type.upper()
        if not rule_type or not name:
            self.module.fail_json(msg='Rule identifier parameters missing')
        return dict(RuleGroupName=name, Type=rule_type)

    @staticmethod
    def _empty_rule_variables():
        return dict(IPSets=dict(), PortSets=dict())

    @staticmethod
    def _transform_rule_variables(variables):
        return {k: dict(Definition=_string_list(v)) for k, v in variables.items()}

    def delete(self, name=None, rule_type=None, arn=None):
        id_params = self._get_id_params(name=name, rule_type=rule_type, arn=arn)
        result = self._get_rule_group(**id_params)
        if not result:
            return False
        self.updated_resource = dict()
        rule_status = self._get_metadata_value('RuleGroupStatus', '').upper()
        if rule_status == 'DELETING':
            self._wait_for_deletion()
            return False
        if self.module.check_mode:
            self.changed = True
            return True
        result = self._delete_rule_group(**id_params)
        self._wait_for_deletion()
        self.changed |= bool(result)
        return bool(result)

    def list(self, scope=None):
        params = dict()
        if scope:
            scope = scope.upper()
            params['Scope'] = scope
        rule_groups = self._list_rule_groups(**params)
        if not rule_groups:
            return list()
        return [r.get('Arn', None) for r in rule_groups]

    def _normalize_rule_variable(self, variable):
        if variable is None:
            return None
        return {k: variable.get(k, dict()).get('Definition', []) for k in variable.keys()}

    def _normalize_rule_variables(self, variables):
        if variables is None:
            return None
        result = dict()
        ip_sets = self._normalize_rule_variable(variables.get('IPSets', None))
        if ip_sets:
            result['ip_sets'] = ip_sets
        port_sets = self._normalize_rule_variable(variables.get('PortSets', None))
        if port_sets:
            result['port_sets'] = port_sets
        return result

    def _normalize_rule_group(self, rule_group):
        if rule_group is None:
            return None
        rule_variables = self._normalize_rule_variables(rule_group.get('RuleVariables', None))
        rule_group = self._normalize_boto3_resource(rule_group)
        if rule_variables is not None:
            rule_group['rule_variables'] = rule_variables
        return rule_group

    def _normalize_rule_group_metadata(self, rule_group_metadata):
        return self._normalize_boto3_resource(rule_group_metadata, add_tags=True)

    def _normalize_rule_group_result(self, result):
        if result is None:
            return None
        rule_group = self._normalize_rule_group(result.get('RuleGroup', None))
        rule_group_metadata = self._normalize_rule_group_metadata(result.get('RuleGroupMetadata', None))
        result = camel_dict_to_snake_dict(result)
        if rule_group:
            result['rule_group'] = rule_group
        if rule_group_metadata:
            result['rule_group_metadata'] = rule_group_metadata
        return result

    def _normalize_resource(self, resource):
        return self._normalize_rule_group_result(resource)

    def get_rule_group(self, name=None, rule_type=None, arn=None):
        id_params = self._get_id_params(name=name, rule_type=rule_type, arn=arn)
        result = self._get_rule_group(**id_params)
        if not result:
            return None
        rule_group = self._normalize_rule_group_result(result)
        return rule_group

    def set_description(self, description):
        return self._set_metadata_value('Description', description)

    def set_capacity(self, capacity):
        return self._set_metadata_value('Capacity', capacity, description='Reserved Capacity', immutable=True)

    def _set_rule_option(self, option_name, description, value, immutable=False, default_value=None):
        if value is None:
            return False
        rule_options = deepcopy(self._get_resource_value('StatefulRuleOptions', dict()))
        if value == rule_options.get(option_name, default_value):
            return False
        if immutable and self.original_resource:
            self.module.fail_json(msg=f'{description} can not be updated after creation')
        rule_options[option_name] = value
        return self._set_resource_value('StatefulRuleOptions', rule_options)

    def set_rule_order(self, order):
        RULE_ORDER_MAP = {'default': 'DEFAULT_ACTION_ORDER', 'strict': 'STRICT_ORDER'}
        value = RULE_ORDER_MAP.get(order)
        changed = self._set_rule_option('RuleOrder', 'Rule order', value, True, 'DEFAULT_ACTION_ORDER')
        self.changed |= changed
        return changed

    def _set_rule_variables(self, set_name, variables, purge):
        if variables is None:
            return False
        variables = self._transform_rule_variables(variables)
        all_variables = deepcopy(self._get_resource_value('RuleVariables', self._empty_rule_variables()))
        current_variables = all_variables.get(set_name, dict())
        updated_variables = _merge_dict(current_variables, variables, purge)
        if current_variables == updated_variables:
            return False
        all_variables[set_name] = updated_variables
        return self._set_resource_value('RuleVariables', all_variables)

    def set_ip_variables(self, variables, purge):
        return self._set_rule_variables('IPSets', variables, purge)

    def set_port_variables(self, variables, purge):
        return self._set_rule_variables('PortSets', variables, purge)

    def _set_rule_source(self, rule_type, rules):
        if not rules:
            return False
        conflicting_types = self.RULE_TYPES.difference({rule_type})
        rules_source = deepcopy(self._get_resource_value('RulesSource', dict()))
        current_keys = set(rules_source.keys())
        conflicting_rule_type = conflicting_types.intersection(current_keys)
        if conflicting_rule_type:
            self.module.fail_json(f'Unable to add {rule_type} rules, {' and '.join(conflicting_rule_type)} rules already set')
        original_rules = rules_source.get(rule_type, None)
        if rules == original_rules:
            return False
        rules_source[rule_type] = rules
        return self._set_resource_value('RulesSource', rules_source)

    def set_rule_string(self, rule):
        if rule is None:
            return False
        if not rule:
            self.module.fail_json('Rule string must include at least one rule')
        rule = '\n'.join(_string_list(rule))
        return self._set_rule_source('RulesString', rule)

    def set_domain_list(self, options):
        if not options:
            return False
        changed = False
        domain_names = options.get('domain_names')
        home_net = options.get('source_ips', None)
        action = options.get('action')
        filter_http = options.get('filter_http', False)
        filter_https = options.get('filter_https', False)
        if home_net:
            changed |= self.set_ip_variables(dict(HOME_NET=home_net), purge=True)
        else:
            self.set_ip_variables(dict(), purge=True)
        target_types = []
        if filter_http:
            target_types.append('HTTP_HOST')
        if filter_https:
            target_types.append('TLS_SNI')
        if action == 'allow':
            action = 'ALLOWLIST'
        else:
            action = 'DENYLIST'
        rule = dict(Targets=domain_names, TargetTypes=target_types, GeneratedRulesType=action)
        changed |= self._set_rule_source('RulesSourceList', rule)
        return changed

    def _format_rule_options(self, options, sid):
        formatted_options = []
        opt = dict(Keyword=f'sid:{sid}')
        formatted_options.append(opt)
        if options:
            for option in sorted(options.keys()):
                opt = dict(Keyword=option)
                settings = options.get(option)
                if settings:
                    opt['Settings'] = _string_list(settings)
                formatted_options.append(opt)
        return formatted_options

    def _format_stateful_rule(self, rule):
        options = self._format_rule_options(rule.get('rule_options', dict()), rule.get('sid'))
        formatted_rule = dict(Action=rule.get('action').upper(), RuleOptions=options, Header=dict(Protocol=rule.get('protocol').upper(), Source=rule.get('source'), SourcePort=rule.get('source_port'), Direction=rule.get('direction').upper(), Destination=rule.get('destination'), DestinationPort=rule.get('destination_port')))
        return formatted_rule

    def set_rule_list(self, rules):
        if rules is None:
            return False
        if not rules:
            self.module.fail_json(msg='Rule list must include at least one rule')
        formatted_rules = [self._format_stateful_rule(r) for r in rules]
        return self._set_rule_source('StatefulRules', formatted_rules)

    def _do_create_resource(self):
        metadata, resource = self._merge_changes(filter_metadata=False)
        params = metadata
        params.update(self._get_id_params())
        params['RuleGroup'] = resource
        response = self._create_rule_group(**params)
        return bool(response)

    def _generate_updated_resource(self):
        metadata, resource = self._merge_changes(filter_metadata=False)
        metadata.update(self._get_id_params())
        updated_resource = dict(RuleGroup=resource, RuleGroupMetadata=metadata)
        return updated_resource

    def _flush_create(self):
        if 'Capacity' not in self._metadata_updates:
            self.module.fail_json('Capacity must be provided when creating a new Rule Group')
        rules_source = self._get_resource_value('RulesSource', dict())
        rule_type = self.RULE_TYPES.intersection(set(rules_source.keys()))
        if len(rule_type) != 1:
            self.module.fail_json('Exactly one of rule strings, domain list or rule list must be provided when creating a new rule group', rule_type=rule_type, keys=self._resource_updates.keys(), types=self.RULE_TYPES)
        return super(NetworkFirewallRuleManager, self)._flush_create()

    def _do_update_resource(self):
        filtered_metadata_updates = self._filter_immutable_metadata_attributes(self._metadata_updates)
        filtered_resource_updates = self._resource_updates
        if not filtered_resource_updates and (not filtered_metadata_updates):
            return False
        metadata, resource = self._merge_changes()
        params = metadata
        params.update(self._get_id_params())
        params['RuleGroup'] = resource
        if not self.module.check_mode:
            response = self._update_rule_group(**params)
        return True

    def _flush_update(self):
        changed = False
        changed |= self._flush_tagging()
        changed |= super(NetworkFirewallRuleManager, self)._flush_update()
        return changed

    def _get_rule_group(self, **params):
        result = self._describe_rule_group(**params)
        if not result:
            return None
        rule_group = result.get('RuleGroup', None)
        metadata = result.get('RuleGroupMetadata', None)
        self._preupdate_resource = deepcopy(rule_group)
        self._preupdate_metadata = deepcopy(metadata)
        return dict(RuleGroup=rule_group, RuleGroupMetadata=metadata)

    def get_resource(self):
        return self.get_rule_group()

    def _do_creation_wait(self, **params):
        all_params = self._get_id_params()
        all_params.update(params)
        return self._wait_rule_group_active(**all_params)

    def _do_deletion_wait(self, **params):
        all_params = self._get_id_params()
        all_params.update(params)
        return self._wait_rule_group_deleted(**all_params)