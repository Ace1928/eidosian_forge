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
def _decode_primary_index(current_table):
    """
    Decodes the primary index info from the current table definition
    splitting it up into the keys we use as parameters
    """
    schema = boto3_tag_list_to_ansible_dict(current_table.get('key_schema', []), tag_name_key_name='key_type', tag_value_key_name='attribute_name')
    attributes = boto3_tag_list_to_ansible_dict(current_table.get('attribute_definitions', []), tag_name_key_name='attribute_name', tag_value_key_name='attribute_type')
    hash_key_name = schema.get('HASH')
    hash_key_type = _short_type_to_long(attributes.get(hash_key_name, None))
    range_key_name = schema.get('RANGE', None)
    range_key_type = _short_type_to_long(attributes.get(range_key_name, None))
    return dict(hash_key_name=hash_key_name, hash_key_type=hash_key_type, range_key_name=range_key_name, range_key_type=range_key_type)