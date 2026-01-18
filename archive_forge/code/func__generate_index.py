from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.dynamodb import wait_indexes_active
from ansible_collections.community.aws.plugins.module_utils.dynamodb import wait_table_exists
from ansible_collections.community.aws.plugins.module_utils.dynamodb import wait_table_not_exists
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _generate_index(index, include_throughput=True):
    key_schema = _generate_schema(index)
    throughput = _generate_throughput(index)
    non_key_attributes = index['includes'] or []
    projection = dict(ProjectionType=index['type'])
    if index['type'] != 'ALL':
        if non_key_attributes:
            projection['NonKeyAttributes'] = non_key_attributes
    elif non_key_attributes:
        module.fail_json(f"DynamoDB does not support specifying non-key-attributes ('includes') for indexes of type 'all'. Index name: {index['name']}")
    idx = dict(IndexName=index['name'], KeySchema=key_schema, Projection=projection)
    if include_throughput:
        idx['ProvisionedThroughput'] = throughput
    return idx