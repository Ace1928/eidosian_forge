from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.base import BaseResourceManager
from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
from ansible_collections.community.aws.plugins.module_utils.base import Boto3Mixin
@AWSRetry.jittered_backoff()
def _paginated_describe_subnets(self, **params):
    paginator = self.client.get_paginator('describe_subnets')
    return paginator.paginate(**params).build_full_result()