from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(catch_extra_error_codes=['InvalidNetworkAclID.NotFound'])
def _create_network_acl_entry(client, *args, **kwargs):
    return client.create_network_acl_entry(*args, **kwargs)