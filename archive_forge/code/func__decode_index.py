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
def _decode_index(index_data, attributes, type_prefix=''):
    try:
        index_map = dict(name=index_data['index_name'])
        index_data = dict(index_data)
        index_data['attribute_definitions'] = attributes
        index_map.update(_decode_primary_index(index_data))
        throughput = index_data.get('provisioned_throughput', {})
        index_map['provisioned_throughput'] = throughput
        if throughput:
            index_map['read_capacity'] = throughput.get('read_capacity_units')
            index_map['write_capacity'] = throughput.get('write_capacity_units')
        projection = index_data.get('projection', {})
        if projection:
            index_map['type'] = type_prefix + projection.get('projection_type')
            index_map['includes'] = projection.get('non_key_attributes', [])
        return index_map
    except Exception as e:
        module.fail_json_aws(e, msg='Decode failure', index_data=index_data)