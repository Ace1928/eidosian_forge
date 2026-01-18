import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class ParameterWaiterFactory(BaseWaiterFactory):

    def __init__(self, module):
        client = module.client('ssm')
        super(ParameterWaiterFactory, self).__init__(module, client)

    @property
    def _waiter_model_data(self):
        data = super(ParameterWaiterFactory, self)._waiter_model_data
        ssm_data = dict(parameter_exists=dict(operation='DescribeParameters', delay=1, maxAttempts=20, acceptors=[dict(state='retry', matcher='error', expected='ParameterNotFound'), dict(state='retry', matcher='path', expected=True, argument='length(Parameters[].Name) == `0`'), dict(state='success', matcher='path', expected=True, argument='length(Parameters[].Name) > `0`')]), parameter_deleted=dict(operation='DescribeParameters', delay=1, maxAttempts=20, acceptors=[dict(state='retry', matcher='path', expected=True, argument='length(Parameters[].Name) > `0`'), dict(state='success', matcher='path', expected=True, argument='length(Parameters[]) == `0`'), dict(state='success', matcher='error', expected='ParameterNotFound')]))
        data.update(ssm_data)
        return data