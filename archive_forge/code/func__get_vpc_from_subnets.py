from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def _get_vpc_from_subnets(self, subnets):
    if not subnets:
        return None
    subnet_details = self._describe_subnets(list(subnets))
    vpc_ids = set((subnet['VpcId'] for subnet in subnet_details))
    if not vpc_ids:
        return None
    if len(vpc_ids) > 1:
        self.module.fail_json('Subnets for an ELB may not span multiple VPCs', subnets=subnet_details, vpc_ids=vpc_ids)
    return vpc_ids.pop()