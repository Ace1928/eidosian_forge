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
def _set_subnets(self):
    """Determine which subnets need to be attached or detached on the ELB"""
    if self.subnets is None:
        return False
    changed = False
    if self.purge_subnets:
        subnets_to_detach = list(set(self.elb['Subnets']) - set(self.subnets))
    else:
        subnets_to_detach = list()
    subnets_to_attach = list(set(self.subnets) - set(self.elb['Subnets']))
    try:
        changed |= self._detach_subnets(subnets_to_detach)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        self.module.fail_json_aws(e, msg='Failed to detach subnets from load balancer')
    try:
        changed |= self._attach_subnets(subnets_to_attach)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        self.module.fail_json_aws(e, msg='Failed to attach subnets to load balancer')
    return changed