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
def _canonicalize_rule_group(self, name, group_type):
    """Iterates through a mixed list of ARNs and Names converting them to
        ARNs.  Also checks that the group type matches the provided group_type.
        """
    arn = None
    if ':' in name:
        arn = name
    else:
        arn = self._rule_group_name_cache.get(name, None)
        if not arn:
            self.module.fail_json('Unable to fetch ARN for rule group', name=name, group_name_cache=self._rule_group_name_cache)
    arn_info = parse_aws_arn(arn)
    if not arn_info:
        self.module.fail_json('Unable to parse ARN for rule group', arn=arn, arn_info=arn_info)
    arn_type = arn_info['resource'].split('/')[0]
    if arn_type != group_type:
        self.module.fail_json('Rule group not of expected type', name=name, arn=arn, expected_type=group_type, found_type=arn_type)
    return arn