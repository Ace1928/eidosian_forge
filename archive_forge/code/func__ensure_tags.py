from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _ensure_tags(redshift, identifier, existing_tags, module):
    """Compares and update resource tags"""
    account_id, partition = get_aws_account_info(module)
    region = module.region
    resource_arn = f'arn:{partition}:redshift:{region}:{account_id}:cluster:{identifier}'
    tags = module.params.get('tags')
    purge_tags = module.params.get('purge_tags')
    tags_to_add, tags_to_remove = compare_aws_tags(boto3_tag_list_to_ansible_dict(existing_tags), tags, purge_tags)
    if tags_to_add:
        try:
            redshift.create_tags(ResourceName=resource_arn, Tags=ansible_dict_to_boto3_tag_list(tags_to_add))
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg='Failed to add tags to cluster')
    if tags_to_remove:
        try:
            redshift.delete_tags(ResourceName=resource_arn, TagKeys=tags_to_remove)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg='Failed to delete tags on cluster')
    changed = bool(tags_to_add or tags_to_remove)
    return changed