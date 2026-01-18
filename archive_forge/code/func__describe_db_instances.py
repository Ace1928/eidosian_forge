from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
@AWSRetry.jittered_backoff()
def _describe_db_instances(conn, **params):
    paginator = conn.get_paginator('describe_db_instances')
    try:
        results = paginator.paginate(**params).build_full_result()['DBInstances']
    except is_boto3_error_code('DBInstanceNotFound'):
        results = []
    return results