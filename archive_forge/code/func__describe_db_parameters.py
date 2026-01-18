from itertools import zip_longest
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
@AWSRetry.jittered_backoff()
def _describe_db_parameters(connection, **params):
    try:
        paginator = connection.get_paginator('describe_db_parameters')
        return paginator.paginate(**params).build_full_result()
    except is_boto3_error_code('DBParameterGroupNotFound'):
        return None