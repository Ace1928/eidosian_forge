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
class NFPolicyBoto3Mixin(NetworkFirewallBoto3Mixin):

    @AWSRetry.jittered_backoff()
    def _paginated_list_policies(self, **params):
        paginator = self.client.get_paginator('list_firewall_policies')
        result = paginator.paginate(**params).build_full_result()
        return result.get('FirewallPolicies', None)

    @Boto3Mixin.aws_error_handler('list all firewall policies')
    def _list_policies(self, **params):
        return self._paginated_list_policies(**params)

    @Boto3Mixin.aws_error_handler('describe firewall policy')
    def _describe_policy(self, **params):
        try:
            result = self.client.describe_firewall_policy(aws_retry=True, **params)
        except is_boto3_error_code('ResourceNotFoundException'):
            return None
        update_token = result.get('UpdateToken', None)
        if update_token:
            self._update_token = update_token
        policy = result.get('FirewallPolicy', None)
        metadata = result.get('FirewallPolicyResponse', None)
        return dict(FirewallPolicy=policy, FirewallPolicyMetadata=metadata)

    @Boto3Mixin.aws_error_handler('create firewall policy')
    def _create_policy(self, **params):
        result = self.client.create_firewall_policy(aws_retry=True, **params)
        update_token = result.get('UpdateToken', None)
        if update_token:
            self._update_token = update_token
        return result.get('FirewallPolicyResponse', None)

    @Boto3Mixin.aws_error_handler('update firewall policy')
    def _update_policy(self, **params):
        if self._update_token and 'UpdateToken' not in params:
            params['UpdateToken'] = self._update_token
        result = self.client.update_firewall_policy(aws_retry=True, **params)
        update_token = result.get('UpdateToken', None)
        if update_token:
            self._update_token = update_token
        return result.get('FirewallPolicyResponse', None)

    @Boto3Mixin.aws_error_handler('delete firewall policy')
    def _delete_policy(self, **params):
        try:
            result = self.client.delete_firewall_policy(aws_retry=True, **params)
        except is_boto3_error_code('ResourceNotFoundException'):
            return None
        return result.get('FirewallPolicyResponse', None)

    @Boto3Mixin.aws_error_handler('firewall policy to finish deleting')
    def _wait_policy_deleted(self, **params):
        waiter = self.nf_waiter_factory.get_waiter('policy_deleted')
        waiter.wait(**params)

    @Boto3Mixin.aws_error_handler('firewall policy to become active')
    def _wait_policy_active(self, **params):
        waiter = self.nf_waiter_factory.get_waiter('policy_active')
        waiter.wait(**params)