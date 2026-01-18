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
class NFFirewallBoto3Mixin(NetworkFirewallBoto3Mixin):

    @AWSRetry.jittered_backoff()
    def _paginated_list_firewalls(self, **params):
        paginator = self.client.get_paginator('list_firewalls')
        result = paginator.paginate(**params).build_full_result()
        return result.get('Firewalls', None)

    @Boto3Mixin.aws_error_handler('list all firewalls')
    def _list_firewalls(self, **params):
        return self._paginated_list_firewalls(**params)

    @Boto3Mixin.aws_error_handler('describe firewall')
    def _describe_firewall(self, **params):
        try:
            result = self.client.describe_firewall(aws_retry=True, **params)
        except is_boto3_error_code('ResourceNotFoundException'):
            return None
        update_token = result.get('UpdateToken', None)
        if update_token:
            self._update_token = update_token
        firewall = result.get('Firewall', None)
        metadata = result.get('FirewallStatus', None)
        return dict(Firewall=firewall, FirewallMetadata=metadata)

    @Boto3Mixin.aws_error_handler('create firewall')
    def _create_firewall(self, **params):
        result = self.client.create_firewall(aws_retry=True, **params)
        update_token = result.get('UpdateToken', None)
        if update_token:
            self._update_token = update_token
        return result.get('FirewallStatus', None)

    @Boto3Mixin.aws_error_handler('update firewall description')
    def _update_firewall_description(self, **params):
        if self._update_token and 'UpdateToken' not in params:
            params['UpdateToken'] = self._update_token
        result = self.client.update_firewall_description(aws_retry=True, **params)
        update_token = result.get('UpdateToken', None)
        if update_token:
            self._update_token = update_token
        return result.get('FirewallName', None)

    @Boto3Mixin.aws_error_handler('update firewall subnet change protection')
    def _update_subnet_change_protection(self, **params):
        if self._update_token and 'UpdateToken' not in params:
            params['UpdateToken'] = self._update_token
        result = self.client.update_subnet_change_protection(aws_retry=True, **params)
        update_token = result.get('UpdateToken', None)
        if update_token:
            self._update_token = update_token
        return result.get('FirewallName', None)

    @Boto3Mixin.aws_error_handler('update firewall policy change protection')
    def _update_firewall_policy_change_protection(self, **params):
        if self._update_token and 'UpdateToken' not in params:
            params['UpdateToken'] = self._update_token
        result = self.client.update_firewall_policy_change_protection(aws_retry=True, **params)
        update_token = result.get('UpdateToken', None)
        if update_token:
            self._update_token = update_token
        return result.get('FirewallName', None)

    @Boto3Mixin.aws_error_handler('update firewall deletion protection')
    def _update_firewall_delete_protection(self, **params):
        if self._update_token and 'UpdateToken' not in params:
            params['UpdateToken'] = self._update_token
        result = self.client.update_firewall_delete_protection(aws_retry=True, **params)
        update_token = result.get('UpdateToken', None)
        if update_token:
            self._update_token = update_token
        return result.get('FirewallName', None)

    @Boto3Mixin.aws_error_handler('associate policy with firewall')
    def _associate_firewall_policy(self, **params):
        if self._update_token and 'UpdateToken' not in params:
            params['UpdateToken'] = self._update_token
        result = self.client.associate_firewall_policy(aws_retry=True, **params)
        update_token = result.get('UpdateToken', None)
        if update_token:
            self._update_token = update_token
        return result.get('FirewallName', None)

    @Boto3Mixin.aws_error_handler('associate subnets with firewall')
    def _associate_subnets(self, **params):
        if self._update_token and 'UpdateToken' not in params:
            params['UpdateToken'] = self._update_token
        result = self.client.associate_subnets(aws_retry=True, **params)
        update_token = result.get('UpdateToken', None)
        if update_token:
            self._update_token = update_token
        return result.get('FirewallName', None)

    @Boto3Mixin.aws_error_handler('disassociate subnets from firewall')
    def _disassociate_subnets(self, **params):
        if self._update_token and 'UpdateToken' not in params:
            params['UpdateToken'] = self._update_token
        result = self.client.disassociate_subnets(aws_retry=True, **params)
        update_token = result.get('UpdateToken', None)
        if update_token:
            self._update_token = update_token
        return result.get('FirewallName', None)

    @Boto3Mixin.aws_error_handler('delete firewall')
    def _delete_firewall(self, **params):
        try:
            result = self.client.delete_firewall(aws_retry=True, **params)
        except is_boto3_error_code('ResourceNotFoundException'):
            return None
        return result.get('FirewallStatus', None)

    @Boto3Mixin.aws_error_handler('firewall to finish deleting')
    def _wait_firewall_deleted(self, **params):
        waiter = self.nf_waiter_factory.get_waiter('firewall_deleted')
        waiter.wait(**params)

    @Boto3Mixin.aws_error_handler('firewall to finish updating')
    def _wait_firewall_updated(self, **params):
        waiter = self.nf_waiter_factory.get_waiter('firewall_updated')
        waiter.wait(**params)

    @Boto3Mixin.aws_error_handler('firewall to become active')
    def _wait_firewall_active(self, **params):
        waiter = self.nf_waiter_factory.get_waiter('firewall_active')
        waiter.wait(**params)