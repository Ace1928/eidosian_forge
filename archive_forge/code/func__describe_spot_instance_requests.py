from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def _describe_spot_instance_requests(connection, **params):
    paginator = connection.get_paginator('describe_spot_instance_requests')
    return paginator.paginate(**params).build_full_result()