from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
@AWSRetry.jittered_backoff(retries=10)
def _describe_global_clusters(client, **params):
    try:
        paginator = client.get_paginator('describe_global_clusters')
        return paginator.paginate(**params).build_full_result()['GlobalClusters']
    except is_boto3_error_code('GlobalClusterNotFoundFault'):
        return []