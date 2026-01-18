import json
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
class UserPolicy(Policy):

    @staticmethod
    def _iam_type():
        return 'user'

    def _list(self, name):
        return self.client.list_user_policies(aws_retry=True, UserName=name)

    def _get(self, name, policy_name):
        return self.client.get_user_policy(aws_retry=True, UserName=name, PolicyName=policy_name)

    def _put(self, name, policy_name, policy_doc):
        return self.client.put_user_policy(aws_retry=True, UserName=name, PolicyName=policy_name, PolicyDocument=policy_doc)

    def _delete(self, name, policy_name):
        return self.client.delete_user_policy(aws_retry=True, UserName=name, PolicyName=policy_name)