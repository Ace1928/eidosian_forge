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
def _global_index_changes(current_table):
    current_global_index_map = current_table['_global_index_map']
    global_index_map = _generate_global_index_map(current_table)
    current_billing_mode = current_table.get('billing_mode')
    if module.params.get('billing_mode') is None:
        billing_mode = current_billing_mode
    else:
        billing_mode = module.params.get('billing_mode')
    include_throughput = True
    if billing_mode == 'PAY_PER_REQUEST':
        include_throughput = False
    index_changes = list()
    for name in global_index_map:
        idx = dict(_generate_index(global_index_map[name], include_throughput=include_throughput))
        if name not in current_global_index_map:
            index_changes.append(dict(Create=idx))
        else:
            _current = current_global_index_map[name]
            _new = global_index_map[name]
            if include_throughput:
                change = dict(_throughput_changes(_current, _new))
                if change:
                    update = dict(IndexName=name, ProvisionedThroughput=change)
                    index_changes.append(dict(Update=update))
    return index_changes