from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import normalize_boto3_result
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
@AWSRetry.jittered_backoff(retries=10)
def _describe_nat_gateways(client, module, **params):
    try:
        paginator = client.get_paginator('describe_nat_gateways')
        return paginator.paginate(**params).build_full_result()['NatGateways']
    except is_boto3_error_code('InvalidNatGatewayID.NotFound'):
        module.exit_json(msg='NAT gateway not found.')
    except is_boto3_error_code('NatGatewayMalformed'):
        module.fail_json_aws(msg='NAT gateway id is malformed.')