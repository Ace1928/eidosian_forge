from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(catch_extra_error_codes=['InvalidClusterState'])
def _modify_cluster(redshift, **kwargs):
    """
    Basic wrapper around modify_cluster with a retry applied.
    Explicitly catches 'InvalidClusterState' (~ Operation in progress) for cases
    where another modification is still in progress
    """
    return redshift.modify_cluster(**kwargs)