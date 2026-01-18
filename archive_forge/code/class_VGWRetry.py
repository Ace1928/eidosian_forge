import time
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class VGWRetry(AWSRetry):

    @staticmethod
    def status_code_from_exception(error):
        return (error.response['Error']['Code'], error.response['Error']['Message'])

    @staticmethod
    def found(response_code, catch_extra_error_codes=None):
        retry_on = ['The maximum number of mutating objects has been reached.']
        if catch_extra_error_codes:
            retry_on.extend(catch_extra_error_codes)
        if not isinstance(response_code, tuple):
            response_code = (response_code,)
        for code in response_code:
            if super().found(response_code, catch_extra_error_codes):
                return True
        return False