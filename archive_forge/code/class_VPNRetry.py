from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class VPNRetry(AWSRetry):

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