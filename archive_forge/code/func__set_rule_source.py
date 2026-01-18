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